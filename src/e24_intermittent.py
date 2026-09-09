"""e24: ゼロ率「中」帯(5-30%)の誤差集中を定量化し、two-part model の改善余地を oracle で測る。

e23 の発見: ゼロ率 5-30% の系列は RMSLE 0.645（全体 0.384 の1.7倍）。
「普段は売れるがたまに売れない」= 間欠需要。RMSLE では 0 と正値のどちらを出すかで大きく損する。

測るもの:
 1. 各帯の SSE 寄与（どれだけ全体を押し下げているか）
 2. oracle A「その行が 0 かどうかを完璧に知っている」場合の RMSLE = two-part model の上限
 3. oracle B「0 の行だけ 0 を出す」場合（= 分類が完璧なら現行予測をどう直すか）
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv2
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
TRAIN_START, HL = pd.Timestamp("2014-04-01"), 365
df = F.build()

zr = df[(df.date >= "2017-01-01") & (df.date < "2017-07-31")].groupby(["store_nbr", "family"], observed=True)["y"].apply(lambda s: (s <= 1e-9).mean())
band = pd.cut(zr, [-.01, .05, .3, .7, 1.01], labels=["密", "中", "疎", "ほぼ0"])

rows = []
for fl, vs, ve, drop in cv2.FOLDS:
    m_tr = (df.date >= TRAIN_START) & (df.date < vs) & df.sales.notna()
    m_va = (df.date >= vs) & (df.date <= ve) & (~df.store_nbr.isin(drop) if drop else True)
    w = 0.5 ** ((vs - df.loc[m_tr, "date"]).dt.days.values / HL)
    ps = []
    for sd in (42, 1):
        mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                        lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"], weight=w,
                                    categorical_feature=F.CATS), num_boost_round=900)
        ps.append(np.clip(mdl.predict(df.loc[m_va, F.FEATURES]), 0, None))
    v = df.loc[m_va, ["store_nbr", "family", "y"]].copy()
    v["pred"] = np.mean(ps, axis=0)
    v["band"] = pd.MultiIndex.from_arrays([v.store_nbr, v.family]).map(band).astype(str)
    v["fold"] = fl
    rows.append(v)
V = pd.concat(rows, ignore_index=True)
V["se"] = (V.pred - V.y) ** 2
base = float(np.sqrt(V.se.mean()))
print(f"ベース RMSLE（2fold pooled, 2seed）{base:.4f}\n")

print("== ゼロ率帯別: 行数 / RMSLE / SSE寄与 / 実際にy=0の割合 ==")
t = V.groupby("band").agg(n=("se", "size"), sse=("se", "sum"), rmsle=("se", lambda x: np.sqrt(x.mean())),
                          zero_rate=("y", lambda s: (s <= 1e-9).mean()), mean_pred=("pred", "mean"))
t["行比"] = t.n / t.n.sum(); t["SSE比"] = t.sse / t.sse.sum()
print(t[["n", "行比", "rmsle", "SSE比", "zero_rate", "mean_pred"]].round(4).to_string())

print("\n== oracle: その行が y==0 かどうかを完璧に知っている場合 ==")
for label, p in [
    ("A: y==0 の行を 0 にする", np.where(V.y.values <= 1e-9, 0.0, V.pred.values)),
    ("B: y>0 の行だけ現行予測、y==0 は 0", np.where(V.y.values <= 1e-9, 0.0, V.pred.values)),
    ("C: y==0 を 0、y>0 を真値（完全oracle）", V.y.values),
]:
    r = float(np.sqrt(np.mean((np.clip(p, 0, None) - V.y.values) ** 2)))
    print(f"  {label:34s} {r:.4f}  ({r-base:+.4f})")

print("\n== 帯別 oracle A の内訳（y==0 の行を 0 にしたときの帯別 RMSLE）==")
V["p_oracle"] = np.where(V.y.values <= 1e-9, 0.0, V.pred.values)
V["se_o"] = (V.p_oracle - V.y) ** 2
o = V.groupby("band").agg(rmsle=("se", lambda x: np.sqrt(x.mean())), rmsle_oracle=("se_o", lambda x: np.sqrt(x.mean())))
o["改善"] = o.rmsle_oracle - o.rmsle
print(o.round(4).to_string())

print("\n== 中帯で、現行モデルは y==0 の行に何を予測しているか ==")
mid = V[V.band == "中"]
z = mid[mid.y <= 1e-9]; nz = mid[mid.y > 1e-9]
print(f"  y==0 の行 {len(z):,}（中帯の {len(z)/len(mid)*100:.1f}%）: 予測平均 {z.pred.mean():.3f}  SSE寄与 {z.se.sum()/mid.se.sum()*100:.1f}%")
print(f"  y>0  の行 {len(nz):,}: 予測平均 {nz.pred.mean():.3f} vs 真値平均 {nz.y.mean():.3f}  SSE寄与 {nz.se.sum()/mid.se.sum()*100:.1f}%")
