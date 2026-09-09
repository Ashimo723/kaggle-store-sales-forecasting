"""e11: CV-LB ギャップ(+0.040)の診断 — 「季節位置を合わせた fold」で測り直す。

e02 の CV fold は 2017-05-28..08-15 の5ブロックで、**8月後半を一度も含んでいない**。
test は 8/16-8/31。過去年の同じ日付窓(08-16..08-31)を valid にすれば、
LB により近い条件での性能が測れる。ギャップが季節位置で説明できるかを見る。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F
from config import PARAMS as _P
PARAMS = {**_P, "num_threads": 6}
TRAIN_START = pd.Timestamp("2013-01-01")  # 過去年の fold でも学習量を確保する

# (ラベル, valid開始, valid終了)
CASES = []
for yr in [2014, 2015, 2016]:
    CASES.append((f"{yr}-08後半(testと同季節)", pd.Timestamp(f"{yr}-08-16"), pd.Timestamp(f"{yr}-08-31")))
    CASES.append((f"{yr}-08前半(対照)",        pd.Timestamp(f"{yr}-07-31"), pd.Timestamp(f"{yr}-08-15")))
    CASES.append((f"{yr}-06後半(対照)",        pd.Timestamp(f"{yr}-06-16"), pd.Timestamp(f"{yr}-06-30")))

df = F.build()
rows = []
for label, vs, ve in CASES:
    t0 = time.time()
    m_tr = (df.date >= TRAIN_START) & (df.date < vs) & df.sales.notna()
    m_va = (df.date >= vs) & (df.date <= ve)
    if m_tr.sum() < 200000:
        print(f"{label:28s} skip (学習データ不足 {m_tr.sum():,})"); continue
    ds = lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"], categorical_feature=F.CATS)
    mdl = lgb.train(PARAMS, ds, num_boost_round=900)  # e08 で iter 依存はほぼ無いと確認済み
    p = np.clip(mdl.predict(df.loc[m_va, F.FEATURES]), 0, None)
    yv = df.loc[m_va, "y"].values
    s = float(np.sqrt(np.mean((p - yv) ** 2)))
    # SCHOOL の寄与も見る（8月は新学期ピーク）
    fam = df.loc[m_va, "family"].values
    sch = fam == "SCHOOL AND OFFICE SUPPLIES"
    se = (p - yv) ** 2
    rows.append(dict(label=label, rmsle=s, n_tr=int(m_tr.sum()),
                     school_rmsle=float(np.sqrt(se[sch].mean())), school_sse=float(se[sch].sum()/se.sum())))
    print(f"{label:28s} RMSLE {s:.4f}  (SCHOOL {rows[-1]['school_rmsle']:.3f}, SSE比 {rows[-1]['school_sse']:.3f})  n_tr {m_tr.sum():,}  {time.time()-t0:.0f}s", flush=True)

r = pd.DataFrame(rows)
print("\n== 季節位置別の平均 RMSLE ==")
for kind in ["08後半", "08前半", "06後半"]:
    sub = r[r.label.str.contains(kind)]
    print(f"  {kind}  {sub.rmsle.mean():.4f}  (年別 {[round(x,4) for x in sub.rmsle]})")
print(f"\n参考: 2017年の通常 fold CV = 0.3856 / 実 Public LB = 0.42567（差 +0.0401）")
