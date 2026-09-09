"""e27: CV 再設計の検証。27 fold の rolling origin で、LB で逆転した3変更を測り直す。

★判定は「同一 fold での対比較」で行う（fold 間変動 0.02 を相殺するため）。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv3
from config import PARAMS as _P
P = {**_P, "num_threads": 8}

SETTINGS = [
    ("e02相当: 2015-01/重みなし", pd.Timestamp("2015-01-01"), 0),
    ("2014-04/重みなし",          pd.Timestamp("2014-04-01"), 0),
    ("2014-04/hl=365",           pd.Timestamp("2014-04-01"), 365),
    ("2015-01/hl=365",           pd.Timestamp("2015-01-01"), 365),
]
df = F.build()
rec = {lab: {} for lab, _, _ in SETTINGS}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    m_va = (df.date >= vs) & (df.date <= ve)
    yv = df.loc[m_va, "y"].values
    for lab, st, hl in SETTINGS:
        m_tr = (df.date >= st) & (df.date < vs) & df.sales.notna()
        w = 0.5 ** ((vs - df.loc[m_tr, "date"]).dt.days.values / hl) if hl else None
        mdl = lgb.train(P, lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"], weight=w,
                                       categorical_feature=F.CATS), num_boost_round=500)
        p = np.clip(mdl.predict(df.loc[m_va, F.FEATURES]), 0, None)
        rec[lab][fname] = float(np.sqrt(np.mean((p - yv) ** 2)))
    print(f"{fname}  " + "  ".join(f"{lab.split(':')[-1][:14]} {rec[lab][fname]:.4f}" for lab, _, _ in SETTINGS) + f"  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec)
R.to_csv(c.OUTPUT / "e27_rolling_cv.csv")
base = SETTINGS[0][0]
print(f"\n== 27 fold 平均 ==")
print(R.mean().round(4).to_string())
print(f"\n== 同一 fold での対比較（{base} 基準）==")
for lab, _, _ in SETTINGS[1:]:
    d = R[lab] - R[base]
    print(f"  {lab:22s} 平均差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():.0f}% ({(d<0).sum()}/{len(d)})  対差sd {d.std():.4f}  t {d.mean()/(d.std()/np.sqrt(len(d))):.2f}")
print(f"\n== 各 fold 単体のばらつき（旧CVの問題の確認）==")
print(f"  fold 間 sd: {R[base].std():.4f}  範囲 {R[base].min():.4f}-{R[base].max():.4f}")
print(f"  2fold の標準誤差 = {R[base].std()/np.sqrt(2):.4f} / 27fold = {R[base].std()/np.sqrt(27):.4f}")
print(f"\n== 直近6 fold（2017-04 以降）だけの対比較 ==")
recent = R.tail(6)
for lab, _, _ in SETTINGS[1:]:
    d = recent[lab] - recent[base]
    print(f"  {lab:22s} 平均差 {d.mean():+.4f}  勝率 {100*(d<0).mean():.0f}% ({(d<0).sum()}/{len(d)})")
