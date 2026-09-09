"""e45: 再帰予測 vs 現行の direct 方式（cv3 27fold 対比較）。

公開ノート(LB 0.37984)の中核。1日先モデルを16回再帰適用する。
h=1 は前日(lag1)が使えるが、h>=2 は予測値を積み上げるので誤差伝播がある。
そのトレードオフを実測する。zero_fc_window=21（公開ノートの設定）も同時に測る。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, cv3, recursive as R
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
ST, SEEDS, NR = pd.Timestamp("2015-01-01"), [42, 1, 2], 500

df = R.build()
keys = (df[["store_nbr", "family"]].drop_duplicates()
        .sort_values(["store_nbr", "family"]).apply(tuple, axis=1).tolist())
keys = pd.MultiIndex.from_tuples(keys, names=["store_nbr", "family"])

rec = {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    origin = vs - pd.Timedelta(days=1)
    m_tr = (df.date >= ST) & (df.date <= origin) & df.sales.notna()
    if m_tr.sum() < 100000: continue
    models = [lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                        lgb.Dataset(df.loc[m_tr, R.FEATURES], df.loc[m_tr, "y"],
                                    categorical_feature=R.CATS), num_boost_round=NR) for sd in SEEDS]
    for zw, tag in [(0, "rec"), (21, "rec_z21")]:
        preds, _ = R.recursive_predict(models, df, keys, origin, 16, zero_fc_window=zw)
        # 評価: 真値と突き合わせ
        rows = []
        for d, p in preds.items():
            rows.append(pd.DataFrame({"store_nbr": keys.get_level_values(0),
                                      "family": keys.get_level_values(1), "date": d, "pred": p}))
        PR = pd.concat(rows, ignore_index=True)
        tv = df[(df.date >= vs) & (df.date <= ve) & df.sales.notna()][["store_nbr", "family", "date", "y"]]
        mg = tv.merge(PR, on=["store_nbr", "family", "date"], how="left")
        rec.setdefault(fname, {})[tag] = float(np.sqrt(np.nanmean((mg.pred - mg.y) ** 2)))
        if tag == "rec":
            mg["h"] = (mg.date - origin).dt.days
            rec[fname]["rec_h1_4"] = float(np.sqrt(np.nanmean((mg[mg.h <= 4].pred - mg[mg.h <= 4].y) ** 2)))
            rec[fname]["rec_h13_16"] = float(np.sqrt(np.nanmean((mg[mg.h >= 13].pred - mg[mg.h >= 13].y) ** 2)))
    # direct（現行 base）を e39 の npz から取得
    p39 = c.OUTPUT / f"e39_pred_{fname}.npz"
    if p39.exists():
        a = np.load(p39)
        rec[fname]["direct"] = float(np.sqrt(np.mean((a["base"] - a["y"]) ** 2)))
        cur = 0.4*a["base"] + 0.2*a["fam"]
        b = np.load(c.OUTPUT / f"e40_pred_{fname}.npz"); g = np.load(c.OUTPUT / f"e42_gseas_{fname}.npz")
        rec[fname]["best4"] = float(np.sqrt(np.mean((cur + 0.2*b["famgrp"] + 0.2*g["p"] - a["y"]) ** 2)))
    print(f"{fname}  direct {rec[fname].get('direct', float('nan')):.4f}  rec {rec[fname]['rec']:.4f}  "
          f"rec_z21 {rec[fname]['rec_z21']:.4f}  (h1-4 {rec[fname]['rec_h1_4']:.4f} / h13-16 {rec[fname]['rec_h13_16']:.4f})  {time.time()-t0:.0f}s", flush=True)

Rd = pd.DataFrame(rec).T
Rd.to_csv(c.OUTPUT / "e45_recursive.csv")
print(f"\n== {len(Rd)} fold 平均 ==")
print(Rd.mean().round(4).to_string())
NY = ["2016-12-26"]
for label, S in [("年末年始を除く", Rd.drop(index=[i for i in Rd.index if i in NY])),
                 ("2017年のみ", Rd[[str(i).startswith("2017") for i in Rd.index]])]:
    for ref in ["direct", "best4"]:
        if ref not in Rd.columns: continue
        print(f"\n== 基準 {ref} / {label} n={len(S)} ==")
        for col in ["rec", "rec_z21"]:
            d = (S[col] - S[ref]).dropna(); se = d.std()/np.sqrt(len(d)); t = d.mean()/se
            ok = "★" if (100*(d<0).mean() >= 85 and d.median() <= d.mean()) or t <= -3 else ""
            print(f"  {col:9s} 差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}%  t {t:+.2f} {ok}")
