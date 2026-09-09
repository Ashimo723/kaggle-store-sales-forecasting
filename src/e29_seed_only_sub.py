"""e29: e02 との差を seed アンサンブルだけにした提出ファイルを作る。
構成 = 56特徴 / 学習開始 2015-01-01 / 重みなし / 3seed / iter 900
cv3 対比較: -0.0020, 勝率 85%(23/27), t=-3.68（現時点で最強の証拠）
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
df = F.build()
m_tr = (df.date >= pd.Timestamp("2015-01-01")) & (~df.is_test) & df.sales.notna()
m_te = df.is_test
print(f"学習 {m_tr.sum():,} 行（e02 と同一条件・重みなし）", flush=True)
ps = []
for sd in [42, 1, 2]:
    t0 = time.time()
    mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                    lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"],
                                categorical_feature=F.CATS), num_boost_round=900)
    ps.append(np.clip(mdl.predict(df.loc[m_te, F.FEATURES]), 0, None))
    print(f"  seed {sd} {time.time()-t0:.0f}s", flush=True)
sub = df.loc[m_te, ["id"]].copy(); sub["sales"] = np.expm1(np.mean(ps, axis=0))
te = pd.read_csv(c.DATA / "test.csv")
c.make_submission(te, sub.sort_values("id").set_index("id").loc[te.id, "sales"].values, "e29_seed3")
for name, lb in [("e02_lgbm", 0.42567), ("e26_best", 0.44717)]:
    prev = pd.read_csv(c.OUTPUT / f"{name}.csv"); cur = pd.read_csv(c.OUTPUT / "e29_seed3.csv")
    d = np.log1p(cur.sales.values) - np.log1p(prev.sales.values)
    print(f"  vs {name}(LB {lb}): log1p RMS {np.sqrt((d**2).mean()):.4f}  平均 {d.mean():+.4f}")
