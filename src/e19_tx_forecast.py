"""e19 (step1): transactions を16日先予測する。sales モデルへ渡す tx_pred を作る。

★train/deploy 整合が最重要:
  train 行に実測 transactions、test 行に予測 transactions を使うと、学習時と配備時で
  特徴の分布が変わり、CV では検出できない形で壊れる（モデル出力由来特徴の典型的な罠）。
  → **全期間について「16日ブロックの rolling origin 予測」**を作り、train 行も test 行も
    同じ「16日先予測」という条件に揃える。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F

P = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=63,
         min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
         lambda_l2=1.0, num_threads=6, verbose=-1, seed=42)
LAGS = [16, 17, 18, 21, 28, 35, 42, 56, 364]
ROLL = [7, 28, 112, 365]


def build_tx_panel():
    """store x date の完全グリッド。sales パネルから店舗単位のカレンダー/祝日/promo を借りる。"""
    df = F.build()
    # store×date 単位に集約（family 方向を潰す）
    cols = ["dow", "day", "month", "year", "dayofyear", "weekofyear", "is_weekend", "is_payday",
            "days_since_payday", "days_to_payday", "is_nye", "hol_national", "hol_regional",
            "hol_local", "hol_any", "is_event", "is_workday", "dcoilwtico", "oil_ma28",
            "city", "state", "type", "cluster"]
    g = df.groupby(["store_nbr", "date"], observed=True)
    t = g[cols].first().reset_index()
    t["promo_store_day"] = g["onpromotion"].sum().values.astype("float32")   # test でも既知
    t["store_open"] = (g["sales"].sum().values > 0).astype("int8")           # 学習行の絞り込み用
    raw = c.load_raw()
    tx = raw["transactions"].copy()
    tx["store_nbr"] = tx.store_nbr.astype(int)
    t["store_nbr"] = t.store_nbr.astype(int)
    t = t.merge(tx, on=["store_nbr", "date"], how="left")
    t["ty"] = np.log1p(t["transactions"]).astype("float32")
    t = t.sort_values(["store_nbr", "date"], ignore_index=True)
    gt = t.groupby("store_nbr", observed=True)["ty"]
    for l in LAGS:
        t[f"tlag{l}"] = gt.shift(l).astype("float32")
    base = gt.shift(16)
    gb = base.groupby(t.store_nbr, observed=True)
    for w in ROLL:
        t[f"trm{w}"] = gb.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean()).astype("float32")
    t["tdow4"] = t[["tlag21", "tlag28", "tlag35", "tlag42"]].mean(axis=1).astype("float32")
    for col in ["city", "state", "type"]:
        t[col] = t[col].astype("category")
    t["store_nbr"] = t["store_nbr"].astype("category")
    t["cluster"] = t["cluster"].astype("category")
    return t


FEATS = (["store_nbr", "city", "state", "type", "cluster"]
         + ["dow", "day", "month", "year", "dayofyear", "weekofyear", "is_weekend", "is_payday",
            "days_since_payday", "days_to_payday", "is_nye"]
         + ["hol_national", "hol_regional", "hol_local", "hol_any", "is_event", "is_workday"]
         + ["dcoilwtico", "oil_ma28", "promo_store_day"]
         + [f"tlag{l}" for l in LAGS] + [f"trm{w}" for w in ROLL] + ["tdow4"])
CATS = ["store_nbr", "city", "state", "type", "cluster"]

if __name__ == "__main__":
    t0 = time.time()
    t = build_tx_panel()
    print(f"tx パネル {t.shape}  実測ありの行 {t.ty.notna().sum():,}", flush=True)

    # rolling origin: 16日ブロックごとに「ブロック開始前まで」で学習し、そのブロックを予測
    starts = pd.date_range("2014-06-01", "2017-08-31", freq="16D")
    t["tx_pred"] = np.nan
    for i, bs in enumerate(starts):
        be = bs + pd.Timedelta(days=15)
        m_tr = (t.date >= "2013-01-01") & (t.date < bs) & t.ty.notna() & (t.store_open == 1)
        m_pr = (t.date >= bs) & (t.date <= be)
        if m_tr.sum() < 5000 or m_pr.sum() == 0:
            continue
        w = 0.5 ** ((bs - t.loc[m_tr, "date"]).dt.days.values / 365)
        mdl = lgb.train(P, lgb.Dataset(t.loc[m_tr, FEATS], t.loc[m_tr, "ty"], weight=w,
                                       categorical_feature=CATS), num_boost_round=400)
        t.loc[m_pr, "tx_pred"] = mdl.predict(t.loc[m_pr, FEATS])
    t["tx_pred"] = t["tx_pred"].astype("float32")

    ok = t.ty.notna() & t.tx_pred.notna() & (t.store_open == 1)
    rms = float(np.sqrt(np.mean((t.loc[ok, "tx_pred"] - t.loc[ok, "ty"]) ** 2)))
    dow4 = float(np.sqrt(np.mean((t.loc[ok, "tdow4"] - t.loc[ok, "ty"]) ** 2)))
    lag16 = float(np.sqrt(np.mean((t.loc[ok, "tlag16"] - t.loc[ok, "ty"]) ** 2)))
    print(f"\n== transactions 16日先予測の精度（log1p 空間 RMSE, 2014-06 以降の OOF）==")
    print(f"  GBM 予測      {rms:.4f}")
    print(f"  同曜日4週平均  {dow4:.4f}")
    print(f"  16日前コピー   {lag16:.4f}")
    print(f"  → GBM は素朴解より {dow4-rms:+.4f}")
    print(f"\ntest 期間(2017-08-16..31)の予測: {t[(t.date>='2017-08-16')].tx_pred.notna().sum()} 行  "
          f"平均 {t[(t.date>='2017-08-16')].tx_pred.mean():.3f}（直近実測 8/1-8/15 の平均 {t[(t.date>='2017-08-01')&(t.date<='2017-08-15')].ty.mean():.3f}）")
    t[["store_nbr", "date", "tx_pred", "ty"]].to_parquet(c.OUTPUT / "tx_pred.parquet", index=False)
    print(f"saved output/tx_pred.parquet  {time.time()-t0:.0f}s")
