#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store Sales — 最終構成で提出ファイルを作る（これ1本で完結）。

    python3 submit.py            # 通常実行（約35分）
    python3 submit.py --smoke    # 動作確認用（seed 1本 / iter 100、約3分）

構成: 4本ブレンド  base .40 + family別 .20 + famgrp .20 + gseas .20
      すべて同じ56特徴・同じ LightGBM で、学習データの切り方だけが違う。
      Public LB 0.41533（114位 / 647、上位17.6%）

前提: data/ に Kaggle のコンペデータ（train.csv, test.csv, stores.csv,
      holidays_events.csv, oil.csv, transactions.csv, sample_submission.csv）があること
        kaggle competitions download -c store-sales-time-series-forecasting -p data
        unzip -o 'data/*.zip' -d data
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import common as c          # noqa: E402  RMSLE / 提出書き出し（行数・id順序・NaN・負値を検算）
import features as F        # noqa: E402  56特徴の生成（ホライズン整合 lag>=16 を担保）

TRAIN_START = pd.Timestamp("2015-01-01")
WEIGHTS = {"base": 0.40, "family": 0.20, "famgrp": 0.20, "gseas": 0.20}
PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=128,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, num_threads=8, verbose=-1)


def add_group_keys(df):
    """family を2通りに集約する。基準は 2016年の統計のみ（test の情報は使わない）。
    famgrp = 売上規模3分位 × ゼロ率(>0.3)、gseas = 年次季節性3分位 × ゼロ率(>0.3)"""
    ref = df[(df.date >= "2016-01-01") & (df.date < "2017-01-01")]
    st = ref.groupby("family", observed=True).agg(
        lvl=("y", "mean"), zr=("y", lambda s: (s <= 1e-9).mean()))
    mo = ref.groupby(["family", ref.date.dt.month], observed=True)["y"].mean().unstack()
    st["seas"] = mo.std(axis=1) / (mo.mean(axis=1).abs() + 1e-6)
    q = lambda s, n: pd.qcut(s.rank(method="first"), n, labels=range(n)).astype(int)
    df["famgrp"] = df["family"].map((q(st.lvl, 3) * 2 + (st.zr > .3).astype(int)).to_dict()).astype("int16")
    df["gseas"] = df["family"].map((q(st.seas, 3) * 2 + (st.zr > .3).astype(int)).to_dict()).astype("int16")
    return df


def fit_predict(train, feats, cats, X, params, seeds, n_round):
    """seed を振って学習し、log1p 空間で平均する（指標が log1p 空間の L2 なのでこの空間で平均する）"""
    ps = []
    for sd in seeds:
        model = lgb.train({**params, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                          lgb.Dataset(train[feats], train["y"], categorical_feature=cats),
                          num_boost_round=n_round)
        ps.append(np.clip(model.predict(X[feats]), 0, None))
    return np.mean(ps, axis=0)


def predict_by_group(key, train, test, feats, cats, fallback, params, seeds, n_round):
    """key ごとに独立したモデルを学習して予測する。学習データが無い群は fallback で埋める"""
    pred = np.full(len(test), np.nan)
    test_key = test[key].values
    groups = list(train.groupby(key, observed=True))
    for i, (k, g) in enumerate(groups, 1):
        mask = test_key == k
        if mask.sum() == 0:
            continue
        pred[mask] = fit_predict(g, feats, cats, test[mask], params, seeds, n_round)
        print(f"    {key}: {i}/{len(groups)}", end="\r", flush=True)
    miss = np.isnan(pred)
    if miss.any():
        print(f"\n    {key}: {miss.sum()} 行を base 予測で補完")
        pred[miss] = fallback[miss]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="seed 1本 / iter 100 で動作確認する")
    ap.add_argument("--out", default="submission", help="出力ファイル名（output/<name>.csv）")
    args = ap.parse_args()
    seeds = [42] if args.smoke else [42, 1, 2]
    n_round = 100 if args.smoke else 900

    need = ["train.csv", "test.csv", "stores.csv", "holidays_events.csv", "oil.csv",
            "transactions.csv", "sample_submission.csv"]
    missing = [f for f in need if not (c.DATA / f).exists()]
    if missing:
        sys.exit(f"data/ に次のファイルがありません: {missing}\n"
                 f"  kaggle competitions download -c store-sales-time-series-forecasting -p data\n"
                 f"  unzip -o 'data/*.zip' -d data")

    t0 = time.time()
    print("[1/3] 特徴量を生成（初回は約1分、2回目以降は output/panel.parquet を再利用）", flush=True)
    df = add_group_keys(F.build())
    train = df[(df.date >= TRAIN_START) & (~df.is_test) & df.sales.notna()]
    test = df[df.is_test]
    print(f"      学習 {len(train):,} 行 / 予測 {len(test):,} 行 / 特徴 {len(F.FEATURES)} 個  ({time.time()-t0:.0f}s)")

    feats_nf = [f for f in F.FEATURES if f != "family"]   # family別モデルでは family は定数
    cats_nf = [x for x in F.CATS if x != "family"]

    print(f"[2/3] 4本の脚を学習（seed {len(seeds)}本 x iter {n_round}）", flush=True)
    legs = {}
    legs["base"] = fit_predict(train, F.FEATURES, F.CATS, test, PARAMS, seeds, n_round)
    print(f"    base 完了  ({time.time()-t0:.0f}s)", flush=True)
    for name, key, fs, cs in [("family", "family", feats_nf, cats_nf),
                              ("famgrp", "famgrp", F.FEATURES, F.CATS),
                              ("gseas", "gseas", F.FEATURES, F.CATS)]:
        legs[name] = predict_by_group(key, train, test, fs, cs, legs["base"], PARAMS, seeds, n_round)
        print(f"    {name} 完了  ({time.time()-t0:.0f}s)", flush=True)

    print("[3/3] ブレンドして書き出す", flush=True)
    blend = sum(WEIGHTS[k] * legs[k] for k in WEIGHTS)          # log1p 空間で加重平均
    sales = np.expm1(np.clip(blend, 0, None))                   # 元の空間に戻して 0 クリップ
    sub = test[["id"]].copy()
    sub["sales"] = sales
    ref = pd.read_csv(c.DATA / "test.csv")
    path = c.make_submission(ref, sub.sort_values("id").set_index("id").loc[ref.id, "sales"].values, args.out)

    for name, p in legs.items():
        print(f"      {name:7s} 平均 {np.expm1(p).mean():8.2f}  相関(base) {np.corrcoef(p, legs['base'])[0,1]:.4f}")
    print(f"\n完了 {time.time()-t0:.0f}s → {path}")
    print(f"提出:  kaggle competitions submit -c store-sales-time-series-forecasting -f {path} -m 'blend4'")


if __name__ == "__main__":
    main()
