"""e22: e21 の交絡除去版。学習開始を 2014-04-01 に戻し（tx_pred が NaN の行も学習に含める）、
3 seed で判定する。e21 は tx_pred の有無と学習期間の2ヶ月短縮が交絡していた。

元 docstring: e21 (step2): 予測 transactions を sales モデルの特徴として投入する（2ステップモデル）。

e19 で作った tx_pred は train 行も test 行も「16日先 rolling origin 予測」なので条件が揃っている。
ラグ transactions（e18: +0.0017 の害）とは別物で、こちらは**予測日当日**の客数見込み。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv2
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
TRAIN_START, HL, SEEDS = pd.Timestamp("2014-04-01"), 365, (42, 1, 2)

df = F.build()
tx = pd.read_parquet(c.OUTPUT / "tx_pred.parquet")
tx["store_nbr"] = tx.store_nbr.astype(int)
df["store_nbr_i"] = df.store_nbr.astype(int)
df = df.merge(tx[["store_nbr", "date", "tx_pred"]], left_on=["store_nbr_i", "date"],
              right_on=["store_nbr", "date"], how="left", suffixes=("", "_y")).drop(columns=["store_nbr_y"])

# 派生: 客数の「平常時からの乖離」と、系列売上水準との積
g = df.groupby("store_nbr_i", observed=True)["tx_pred"]
df["tx_pred_ma28"] = g.transform(lambda s: s.rolling(28, min_periods=1).mean()).astype("float32")
df["tx_pred_dev"] = (df["tx_pred"] - df["tx_pred_ma28"]).astype("float32")
df["tx_pred_x_level"] = (df["tx_pred_dev"] * df["rmean7"]).astype("float32")
df["tx_pred"] = df["tx_pred"].astype("float32")

SETS = {
    "ベース56": F.FEATURES,
    "+tx_pred": F.FEATURES + ["tx_pred"],
    "+tx_pred_dev": F.FEATURES + ["tx_pred_dev"],
    "+tx_pred 一式": F.FEATURES + ["tx_pred", "tx_pred_dev", "tx_pred_x_level"],
}
res = {}
for label, feats in SETS.items():
    per, t0 = {}, time.time()
    for fl, vs, ve, drop in cv2.FOLDS:
        m_tr = (df.date >= TRAIN_START) & (df.date < vs) & df.sales.notna()
        m_va = (df.date >= vs) & (df.date <= ve) & (~df.store_nbr.isin(drop) if drop else True)
        w = 0.5 ** ((vs - df.loc[m_tr, "date"]).dt.days.values / HL)
        yv = df.loc[m_va, "y"].values
        ps = []
        for sd in SEEDS:
            mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                            lgb.Dataset(df.loc[m_tr, feats], df.loc[m_tr, "y"], weight=w,
                                        categorical_feature=F.CATS), num_boost_round=900)
            ps.append(np.clip(mdl.predict(df.loc[m_va, feats]), 0, None))
        per[fl] = float(np.sqrt(np.mean((np.mean(ps, axis=0) - yv) ** 2)))
        if label == "+tx_pred 一式" and fl.startswith("A"):
            imp = pd.Series(mdl.feature_importance("gain"), index=feats)
    res[label] = per
    print(f"{label:16s} A {per['A:2017-08前半']:.4f}  B {per['B:2016-08後半']:.4f}  平均 {np.mean(list(per.values())):.4f}  {time.time()-t0:.0f}s", flush=True)

b = res["ベース56"]; bm = np.mean(list(b.values()))
print("\n== ベース比（採用は符号一致かつ |差|>=0.002）==")
for label, per in res.items():
    da, db = per["A:2017-08前半"] - b["A:2017-08前半"], per["B:2016-08後半"] - b["B:2016-08後半"]
    m = np.mean(list(per.values()))
    tag = "★改善" if (da < 0 and db < 0 and abs(m - bm) >= 0.002) else ("同方向だが小" if (da < 0) == (db < 0) else "符号不一致")
    print(f"{label:16s} A {da:+.4f}  B {db:+.4f}  平均 {m-bm:+.4f}  {tag}")
print("\n== tx_pred 系の gain 比 ==")
print((imp / imp.sum())[["tx_pred", "tx_pred_dev", "tx_pred_x_level"]].round(4).to_string())
print("（参考 rmean7: %.4f）" % (imp["rmean7"] / imp.sum()))
