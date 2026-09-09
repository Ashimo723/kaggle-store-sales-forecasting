"""e16: seed アンサンブル。現状は単一 LightGBM 1本のみで、seed 違いすら試していない。
まず最も確実な分散削減を取る。log1p 空間で平均（指標が log1p 空間の L2 なのでこの空間で平均するのが正しい）。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv2
from config import PARAMS as _P
BASE = {**_P, "num_threads": 8}
TRAIN_START, HL, SEEDS = pd.Timestamp("2014-04-01"), 365, [42, 1, 2, 3, 4]

df = F.build()
per_fold = {}
for fl, vs, ve, drop in cv2.FOLDS:
    m_tr = (df.date >= TRAIN_START) & (df.date < vs) & df.sales.notna()
    m_va = (df.date >= vs) & (df.date <= ve) & (~df.store_nbr.isin(drop) if drop else True)
    w = 0.5 ** ((vs - df.loc[m_tr, "date"]).dt.days.values / HL)
    yv = df.loc[m_va, "y"].values
    ps, singles = [], []
    for sd in SEEDS:
        P = {**BASE, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd}
        mdl = lgb.train(P, lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"], weight=w,
                                       categorical_feature=F.CATS), num_boost_round=900)
        p = np.clip(mdl.predict(df.loc[m_va, F.FEATURES]), 0, None)
        ps.append(p); singles.append(float(np.sqrt(np.mean((p - yv) ** 2))))
    ens = [float(np.sqrt(np.mean((np.mean(ps[:k], axis=0) - yv) ** 2))) for k in range(1, len(SEEDS) + 1)]
    per_fold[fl] = dict(singles=singles, ens=ens)
    print(f"{fl}  単体 {[round(x,4) for x in singles]}  → 平均 {[round(x,4) for x in ens]}", flush=True)

print("\n== seed 数別の 2fold 平均 ==")
for k in range(len(SEEDS)):
    m = np.mean([per_fold[f]["ens"][k] for f in per_fold])
    print(f"  {k+1} seed: {m:.4f}" + (f"  (1seed比 {m - np.mean([per_fold[f]['ens'][0] for f in per_fold]):+.4f})" if k else ""))
sd_single = np.std([s for f in per_fold for s in per_fold[f]["singles"]])
print(f"\n単体 seed 間の標準偏差: {sd_single:.4f}（判定ノイズ床 0.002 との比較用）")
