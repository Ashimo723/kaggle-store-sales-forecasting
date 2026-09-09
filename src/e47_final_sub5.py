"""e47: 5本ブレンドの提出ファイル。
base .32 + family別 .16 + famgrp .16 + gseas .16 + 再帰 .20
（= best4(base.4/fam.2/fg.2/gs.2) × 0.8 + 再帰 × 0.2）
cv3: 年末年始除く26fold -0.0029（中央値-0.0026, 勝率92%, t=-4.50）/ 2017年13fold -0.0016（勝率92%）
現 best e44（LB 0.41533）からの変更 = 再帰予測脚の追加のみ。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, recursive as R
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
ST, SEEDS, NR = pd.Timestamp("2015-01-01"), [42, 1, 2], 900
ORIGIN = pd.Timestamp("2017-08-15")

# ---------- (A) 再帰脚 ----------
t0 = time.time()
dr = R.build()
keys = pd.MultiIndex.from_tuples(
    dr[["store_nbr", "family"]].drop_duplicates().sort_values(["store_nbr", "family"]).apply(tuple, axis=1).tolist(),
    names=["store_nbr", "family"])
m_tr = (dr.date >= ST) & (dr.date <= ORIGIN) & dr.sales.notna()
models = [lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                    lgb.Dataset(dr.loc[m_tr, R.FEATURES], dr.loc[m_tr, "y"],
                                categorical_feature=R.CATS), num_boost_round=NR) for sd in SEEDS]
print(f"再帰モデル学習 done {time.time()-t0:.0f}s", flush=True)
preds, _ = R.recursive_predict(models, dr, keys, ORIGIN, 16, zero_fc_window=0)
REC = pd.concat([pd.DataFrame({"store_nbr": keys.get_level_values(0), "family": keys.get_level_values(1),
                               "date": d, "rec": p}) for d, p in preds.items()], ignore_index=True)
print(f"再帰予測 done {time.time()-t0:.0f}s  行数 {len(REC)}", flush=True)

# ---------- (B) 4本の direct 脚 ----------
df = F.build()
ref = df[(df.date >= "2016-01-01") & (df.date < "2017-01-01")]
st = ref.groupby("family", observed=True).agg(lvl=("y", "mean"), zr=("y", lambda s: (s <= 1e-9).mean()))
mo = ref.groupby(["family", ref.date.dt.month], observed=True)["y"].mean().unstack()
st["seas"] = mo.std(axis=1) / (mo.mean(axis=1).abs() + 1e-6)
def q(s, n): return pd.qcut(s.rank(method="first"), n, labels=range(n)).astype(int)
df["fg"] = df["family"].map((q(st.lvl, 3) * 2 + (st.zr > .3).astype(int)).to_dict()).astype("int16")
df["gs"] = df["family"].map((q(st.seas, 3) * 2 + (st.zr > .3).astype(int)).to_dict()).astype("int16")
tr, te = df[(df.date >= ST) & (~df.is_test) & df.sales.notna()], df[df.is_test]
FEATS_NF = [f for f in F.FEATURES if f != "family"]; CATS_NF = [x for x in F.CATS if x != "family"]

def fit_pred(g, feats, cats, X):
    return np.mean([np.clip(lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                                      lgb.Dataset(g[feats], g["y"], categorical_feature=cats),
                                      num_boost_round=NR).predict(X[feats]), 0, None) for sd in SEEDS], axis=0)

base = fit_pred(tr, F.FEATURES, F.CATS, te); print(f"base done {time.time()-t0:.0f}s", flush=True)
def by_key(key, feats, cats):
    p = np.zeros(len(te)); vk = te[key].values
    for k_, g in tr.groupby(key, observed=True):
        m = vk == k_
        if m.sum(): p[m] = fit_pred(g, feats, cats, te[m])
    miss = p == 0
    if miss.any(): p[miss] = base[miss]
    return p
fam = by_key("family", FEATS_NF, CATS_NF); print(f"family done {time.time()-t0:.0f}s", flush=True)
fg = by_key("fg", F.FEATURES, F.CATS);     print(f"famgrp done {time.time()-t0:.0f}s", flush=True)
gs = by_key("gs", F.FEATURES, F.CATS);     print(f"gseas done {time.time()-t0:.0f}s", flush=True)

# ---------- (C) 合成 ----------
out = te[["id", "store_nbr", "family", "date"]].copy()
out["direct4"] = 0.4*base + 0.2*fam + 0.2*fg + 0.2*gs
out = out.merge(REC, on=["store_nbr", "family", "date"], how="left")
print("再帰が付かなかった行:", int(out.rec.isna().sum()))
out["rec"] = out["rec"].fillna(out["direct4"])
out["blend"] = 0.8*out["direct4"] + 0.2*out["rec"]
out["sales"] = np.expm1(np.clip(out["blend"], 0, None))
refcsv = pd.read_csv(c.DATA / "test.csv")
c.make_submission(refcsv, out.sort_values("id").set_index("id").loc[refcsv.id, "sales"].values, "e47_blend5")
prev = pd.read_csv(c.OUTPUT / "e44_blend4.csv"); cur = pd.read_csv(c.OUTPUT / "e47_blend5.csv")
d = np.log1p(cur.sales.values) - np.log1p(prev.sales.values)
print(f"  vs e44(LB 0.41533): log1p RMS {np.sqrt((d**2).mean()):.4f}  平均 {d.mean():+.4f}  |差|>0.1 の行 {100*(np.abs(d)>0.1).mean():.1f}%")
