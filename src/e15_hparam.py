"""e15: ハイパラ探索（新CV 2fold）。容量・正則化・目的関数は一度も振っていない軸。
ベース = 2014-04起点 + hl=365（新CV 0.3828）。判定は 2fold 平均かつ符号一致、差 0.002 未満は判定不能。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv2
from config import PARAMS as _P

BASE = {**_P, "num_threads": 8}
TRAIN_START, HL = pd.Timestamp("2014-04-01"), 365
CASES = [
    ("base (leaves128/min100/lr.05)", {}, 900),
    ("leaves 256",                    {"num_leaves": 256}, 900),
    ("leaves 512",                    {"num_leaves": 512}, 900),
    ("leaves 64",                     {"num_leaves": 64}, 900),
    ("min_data 20",                   {"min_data_in_leaf": 20}, 900),
    ("min_data 300",                  {"min_data_in_leaf": 300}, 900),
    ("lr .03 / iter 1600",            {"learning_rate": 0.03}, 1600),
    ("feature_fraction .5",           {"feature_fraction": 0.5}, 900),
    ("lambda_l2 10",                  {"lambda_l2": 10.0}, 900),
    ("objective huber",               {"objective": "huber"}, 900),
]

df = F.build()
rows = []
for label, over, nround in CASES:
    P = {**BASE, **over}
    per = {}
    t0 = time.time()
    for fl, vs, ve, drop in cv2.FOLDS:
        m_tr = (df.date >= TRAIN_START) & (df.date < vs) & df.sales.notna()
        m_va = (df.date >= vs) & (df.date <= ve) & (~df.store_nbr.isin(drop) if drop else True)
        w = 0.5 ** ((vs - df.loc[m_tr, "date"]).dt.days.values / HL)
        mdl = lgb.train(P, lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"], weight=w,
                                       categorical_feature=F.CATS), num_boost_round=nround)
        p = np.clip(mdl.predict(df.loc[m_va, F.FEATURES]), 0, None)
        per[fl] = float(np.sqrt(np.mean((p - df.loc[m_va, "y"].values) ** 2)))
    rows.append(dict(label=label, **per, mean=np.mean(list(per.values()))))
    print(f"{label:32s} A {per['A:2017-08前半']:.4f}  B {per['B:2016-08後半']:.4f}  平均 {rows[-1]['mean']:.4f}  {time.time()-t0:.0f}s", flush=True)

r = pd.DataFrame(rows); b = r.iloc[0]
print("\n== base 比（負が改善 / 採用は符号一致かつ |差| >= 0.002）==")
for _, x in r.iterrows():
    da, db = x["A:2017-08前半"] - b["A:2017-08前半"], x["B:2016-08後半"] - b["B:2016-08後半"]
    ok = "採用可" if ((da < 0) == (db < 0)) and abs(x["mean"] - b["mean"]) >= 0.002 else ("判定不能" if (da<0)==(db<0) else "符号不一致")
    print(f"{x.label:32s} A {da:+.4f}  B {db:+.4f}  平均 {x['mean']-b['mean']:+.4f}   {ok}")
r.to_csv(c.OUTPUT / "e15_hparam.csv", index=False)
