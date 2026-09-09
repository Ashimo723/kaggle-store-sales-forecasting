"""e28: LB で逆転した2構成を 27 fold で完全対比較する。

  A: 2015-01起点 / 重みなし / 1seed  = 提出#1 e02  → LB 0.42567
  D: 2014-04起点 / hl=365  / 3seed  ≒ 提出#2 e26  → LB 0.44717（+0.0215）

各設定で seed 3本の予測を作り、1seed版と3seed版を同時に評価する（seed アンサンブルの効果を単離）。
判定は同一 fold での対比較（e27: 対差 sd 0.0076-0.0128 → 標準誤差 0.0017 で 0.002 級を判定可能）。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv3
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
SEEDS = [42, 1, 2]
ARMS = [("2015-01/重みなし", pd.Timestamp("2015-01-01"), 0),
        ("2014-04/hl365",   pd.Timestamp("2014-04-01"), 365)]

df = F.build()
rec = {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    m_va = (df.date >= vs) & (df.date <= ve)
    yv = df.loc[m_va, "y"].values
    row = {}
    for arm, st, hl in ARMS:
        m_tr = (df.date >= st) & (df.date < vs) & df.sales.notna()
        w = 0.5 ** ((vs - df.loc[m_tr, "date"]).dt.days.values / hl) if hl else None
        ps = []
        for sd in SEEDS:
            mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                            lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"], weight=w,
                                        categorical_feature=F.CATS), num_boost_round=900)
            ps.append(np.clip(mdl.predict(df.loc[m_va, F.FEATURES]), 0, None))
        row[f"{arm}/1seed"] = float(np.sqrt(np.mean((ps[0] - yv) ** 2)))
        row[f"{arm}/3seed"] = float(np.sqrt(np.mean((np.mean(ps, axis=0) - yv) ** 2)))
    rec[fname] = row
    print(f"{fname}  " + "  ".join(f"{k.split('/')[0][:7]}{k.split('/')[1]} {v:.4f}" for k, v in row.items()) + f"  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec).T
R.to_csv(c.OUTPUT / "e28_lb_pair.csv")
A = "2015-01/重みなし/1seed"; D = "2014-04/hl365/3seed"
print("\n== 27 fold 平均 ==")
print(R.mean().round(4).to_string())
print(f"\n== 同一 fold 対比較（基準 = {A}、これが LB 0.42567 の構成）==")
for col in R.columns:
    if col == A: continue
    d = R[col] - R[A]
    se = d.std() / np.sqrt(len(d))
    print(f"  {col:26s} 平均差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}% ({(d<0).sum()}/{len(d)})  SE {se:.4f}  t {d.mean()/se:+.2f}")
print(f"\n★ LB での実測差（{A} → {D}）= +0.0215")
print(f"★ CV での差 = {(R[D]-R[A]).mean():+.4f}")
print("\n== seed アンサンブル単独の効果（同一起点・同一重みで 1seed → 3seed）==")
for arm, _, _ in ARMS:
    d = R[f"{arm}/3seed"] - R[f"{arm}/1seed"]
    se = d.std() / np.sqrt(len(d))
    print(f"  {arm:16s} 平均差 {d.mean():+.4f}  勝率 {100*(d<0).mean():3.0f}%  t {d.mean()/se:+.2f}")
print("\n== fold 別: D-A の差が大きい順 上位5 / 下位5（どの時期で逆転するか）==")
d = (R[D] - R[A]).sort_values()
print("  D が勝つ:", ", ".join(f"{i}:{v:+.4f}" for i, v in d.head(5).items()))
print("  D が負ける:", ", ".join(f"{i}:{v:+.4f}" for i, v in d.tail(5).items()))
