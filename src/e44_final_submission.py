"""e44: 4本ブレンドの提出ファイル。base .40 + family別 .20 + famgrp .20 + gseas .20
cv3: 年末年始除く26fold で -0.0018(勝率92%, t=-6.17) / 2017年13fold で -0.0012(勝率85%, t=-4.87)
現 best e38（LB 0.41889）からの変更 = famgrp脚と gseas脚の追加（比率も再配分）。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
ST, SEEDS, NR = pd.Timestamp("2015-01-01"), [42, 1, 2], 900
W = dict(base=.40, fam=.20, fg=.20, gs=.20)

df = F.build()
# グループ定義（e42 と同一: 2016年の統計で算出）
ref = df[(df.date >= "2016-01-01") & (df.date < "2017-01-01")]
st = ref.groupby("family", observed=True).agg(lvl=("y", "mean"), zr=("y", lambda s: (s <= 1e-9).mean()))
mo = ref.groupby(["family", ref.date.dt.month], observed=True)["y"].mean().unstack()
st["seas"] = mo.std(axis=1) / (mo.mean(axis=1).abs() + 1e-6)
def q(s, n): return pd.qcut(s.rank(method="first"), n, labels=range(n)).astype(int)
df["fg"] = df["family"].map((q(st.lvl, 3) * 2 + (st.zr > .3).astype(int)).to_dict()).astype("int16")
df["gs"] = df["family"].map((q(st.seas, 3) * 2 + (st.zr > .3).astype(int)).to_dict()).astype("int16")
print("fg 群:", df.fg.nunique(), " gs 群:", df.gs.nunique())

m_tr = (df.date >= ST) & (~df.is_test) & df.sales.notna()
m_te = df.is_test
tr, te = df[m_tr], df[m_te]
FEATS_NF = [f for f in F.FEATURES if f != "family"]
CATS_NF = [x for x in F.CATS if x != "family"]

def fit_pred(g, feats, cats, X):
    ps = []
    for sd in SEEDS:
        m = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                      lgb.Dataset(g[feats], g["y"], categorical_feature=cats), num_boost_round=NR)
        ps.append(np.clip(m.predict(X[feats]), 0, None))
    return np.mean(ps, axis=0)

t0 = time.time()
base = fit_pred(tr, F.FEATURES, F.CATS, te); print(f"base done {time.time()-t0:.0f}s", flush=True)

def by_key(key, feats, cats):
    p = np.zeros(len(te)); vk = te[key].values
    for k_, g in tr.groupby(key, observed=True):
        m = vk == k_
        if m.sum() == 0: continue
        p[m] = fit_pred(g, feats, cats, te[m])
    miss = p == 0
    if miss.any(): p[miss] = base[miss]
    return p

fam = by_key("family", FEATS_NF, CATS_NF); print(f"family別 done {time.time()-t0:.0f}s", flush=True)
fg = by_key("fg", F.FEATURES, F.CATS);     print(f"famgrp done {time.time()-t0:.0f}s", flush=True)
gs = by_key("gs", F.FEATURES, F.CATS);     print(f"gseas done {time.time()-t0:.0f}s", flush=True)

blend = W["base"]*base + W["fam"]*fam + W["fg"]*fg + W["gs"]*gs
sub = te[["id"]].copy(); sub["sales"] = np.expm1(np.clip(blend, 0, None))
refcsv = pd.read_csv(c.DATA / "test.csv")
c.make_submission(refcsv, sub.sort_values("id").set_index("id").loc[refcsv.id, "sales"].values, "e44_blend4")
# 前回提出との差分（存在する場合のみ。クローン直後は無いのでスキップされる）
prev_path = c.OUTPUT / "e38_famblend.csv"
if prev_path.exists():
    prev = pd.read_csv(prev_path); cur = pd.read_csv(c.OUTPUT / "e44_blend4.csv")
    d = np.log1p(cur.sales.values) - np.log1p(prev.sales.values)
    print(f"  vs e38(LB 0.41889): log1p RMS {np.sqrt((d**2).mean()):.4f}  平均 {d.mean():+.4f}  |差|>0.1 の行 {100*(np.abs(d)>0.1).mean():.1f}%")
