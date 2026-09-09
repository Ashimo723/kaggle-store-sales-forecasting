"""e30: cv3(|t|>=3 基準)で、判定不能のまま保留していた施策を測り直す。

対象:
  1. seed を 3 → 5 に増やす（転移実績のある軸を伸ばせるか）
  2. 特徴の下位10カット（cv2 で -0.0006、判定不能だった）
  3. 集約系列の追加（cv2 で +0.0016 だが符号不一致だった）
★valid は sales.notna() で絞る（12/25 欠測日で NaN になるバグの修正）
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, features2 as F2, cv3
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
ST, SEEDS = pd.Timestamp("2015-01-01"), [42, 1, 2, 3, 4]

df = F2.build()   # 56特徴 + 拡張列（集約系列を使うため）
AGG = [x for x in F2.NEW if x.startswith("ag")]

# 下位10特徴（e20 の重要度から）
DROP10 = ["lag18", "lag19", "hol_any", "state", "is_event", "type", "is_payday",
          "hol_local", "hol_regional", "is_workday"]
ARMS = {
    "base(56)":      F.FEATURES,
    "下位10カット(46)": [f for f in F.FEATURES if f not in DROP10],
    "+集約系列(68)":    F.FEATURES + AGG,
}
rec = {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    m_va = (df.date >= vs) & (df.date <= ve) & df.sales.notna()   # ★12/25 欠測を除外
    if m_va.sum() == 0:
        continue
    yv = df.loc[m_va, "y"].values
    m_tr = (df.date >= ST) & (df.date < vs) & df.sales.notna()
    row = {}
    for arm, feats in ARMS.items():
        seeds = SEEDS if arm == "base(56)" else SEEDS[:3]
        ps = []
        for sd in seeds:
            mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                            lgb.Dataset(df.loc[m_tr, feats], df.loc[m_tr, "y"],
                                        categorical_feature=[x for x in F.CATS if x in feats]),
                            num_boost_round=500)
            ps.append(np.clip(mdl.predict(df.loc[m_va, feats]), 0, None))
        row[f"{arm}/3seed"] = float(np.sqrt(np.mean((np.mean(ps[:3], axis=0) - yv) ** 2)))
        if arm == "base(56)":
            row["base(56)/5seed"] = float(np.sqrt(np.mean((np.mean(ps, axis=0) - yv) ** 2)))
    rec[fname] = row
    print(f"{fname}  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()) + f"  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec).T
R.to_csv(c.OUTPUT / "e30_cv3_recheck.csv")
B = "base(56)/3seed"
print(f"\n== {len(R)} fold 平均 ==")
print(R.mean().round(4).to_string())
print(f"\n== 同一 fold 対比較（基準 {B} = 現 best 構成）/ 採用は |t|>=3 ==")
for col in R.columns:
    if col == B: continue
    d = (R[col] - R[B]).dropna()
    se = d.std() / np.sqrt(len(d)); t = d.mean() / se
    print(f"  {col:22s} 平均差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}% ({(d<0).sum()}/{len(d)})  t {t:+.2f}   {'★採用可' if t <= -3 else '不採用'}")
