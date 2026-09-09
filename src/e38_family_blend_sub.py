"""e38: 現 best（3seed）に family別モデルを 30% ブレンドした提出ファイルを作る。

cv3: 年末年始 fold を除く21本で 21/21 改善 t=-7.10 / 2017年の8 fold で 8/8 改善 t=-8.91
     全27 fold でも 21/22 改善（勝率95%、二項検定 p≈1.1e-5）
現 best = e29（2015-01起点・重みなし・3seed、LB 0.42468）からの変更は**この1点のみ**。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
ST, SEEDS, W = pd.Timestamp("2015-01-01"), [42, 1, 2], 0.30
FEATS_NF = [f for f in F.FEATURES if f != "family"]
CATS_NF = [x for x in F.CATS if x != "family"]

df = F.build()
m_tr = (df.date >= ST) & (~df.is_test) & df.sales.notna()
m_te = df.is_test
tr, te = df[m_tr], df[m_te]
print(f"学習 {len(tr):,} 行 / 予測 {len(te):,} 行", flush=True)

ps = []
for sd in SEEDS:
    t0 = time.time()
    mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                    lgb.Dataset(tr[F.FEATURES], tr["y"], categorical_feature=F.CATS), num_boost_round=900)
    ps.append(np.clip(mdl.predict(te[F.FEATURES]), 0, None))
    print(f"  base seed {sd} {time.time()-t0:.0f}s", flush=True)
base = np.mean(ps, axis=0)

pred_fam = np.zeros(len(te)); te_fam = te["family"].values
for i, (fam, g) in enumerate(tr.groupby("family", observed=True)):
    mva = te_fam == fam
    if mva.sum() == 0: continue
    pb = []
    for sd in SEEDS:
        m2 = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                       lgb.Dataset(g[FEATS_NF], g["y"], categorical_feature=CATS_NF), num_boost_round=900)
        pb.append(np.clip(m2.predict(te[mva][FEATS_NF]), 0, None))
    pred_fam[mva] = np.mean(pb, axis=0)
    if i % 8 == 0: print(f"  family {i+1}/33 done", flush=True)

blend = (1 - W) * base + W * pred_fam
sub = te[["id"]].copy(); sub["sales"] = np.expm1(np.clip(blend, 0, None))
ref = pd.read_csv(c.DATA / "test.csv")
c.make_submission(ref, sub.sort_values("id").set_index("id").loc[ref.id, "sales"].values, "e38_famblend")
prev = pd.read_csv(c.OUTPUT / "e29_seed3.csv"); cur = pd.read_csv(c.OUTPUT / "e38_famblend.csv")
d = np.log1p(cur.sales.values) - np.log1p(prev.sales.values)
print(f"  vs e29(LB 0.42468): log1p RMS {np.sqrt((d**2).mean()):.4f}  平均 {d.mean():+.4f}  |差|>0.1 の行 {100*(np.abs(d)>0.1).mean():.1f}%")
