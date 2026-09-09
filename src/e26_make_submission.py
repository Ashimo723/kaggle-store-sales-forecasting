"""e26: 現在の best 構成で提出ファイルを作る（提出はしない）。
構成 = 56特徴 / 学習開始 2014-04-01 / 直近重み hl=365 / 5 seed 平均 / iter 900
新CV 0.3823（e02 の 0.3856 から -0.0033）
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
TRAIN_START, HL, SEEDS = pd.Timestamp("2014-04-01"), 365, [42, 1, 2, 3, 4]

df = F.build()
m_tr = (df.date >= TRAIN_START) & (~df.is_test) & df.sales.notna()
m_te = df.is_test
w = 0.5 ** ((c.TEST_START - df.loc[m_tr, "date"]).dt.days.values / HL)
print(f"学習 {m_tr.sum():,} 行 / 予測 {m_te.sum():,} 行", flush=True)
ps = []
for sd in SEEDS:
    t0 = time.time()
    mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                    lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"], weight=w,
                                categorical_feature=F.CATS), num_boost_round=900)
    ps.append(np.clip(mdl.predict(df.loc[m_te, F.FEATURES]), 0, None))
    print(f"  seed {sd} done {time.time()-t0:.0f}s", flush=True)
pred_log = np.mean(ps, axis=0)
sub = df.loc[m_te, ["id"]].copy()
sub["sales"] = np.expm1(pred_log)
sub = sub.sort_values("id")
te = pd.read_csv(c.DATA / "test.csv")
c.make_submission(te, sub.set_index("id").loc[te.id, "sales"].values, "e26_best",
                  message="e26 lgbm hl365 5seed CV0.3823")
# 前回提出との差分を見る（LB の変化がどこから来るかの手掛かり）
prev = pd.read_csv(c.OUTPUT / "e02_lgbm.csv")
cur = pd.read_csv(c.OUTPUT / "e26_best.csv")
d = np.log1p(cur.sales.values) - np.log1p(prev.sales.values)
print(f"\n前回提出(e02, LB 0.42567)との差: log1p 空間 RMS {np.sqrt((d**2).mean()):.4f}  平均 {d.mean():+.4f}  |差|>0.1 の行 {100*(np.abs(d)>0.1).mean():.1f}%")
