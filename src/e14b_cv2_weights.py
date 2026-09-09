"""e14: 既存施策を新 CV（seasonal 2fold）で測り直す。
旧CV（2017-05-28..08-15 の5fold）での判定が変わるかを見る。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv2
from config import PARAMS as _P
PARAMS = {**_P, "num_threads": 8}

# (ラベル, 学習開始, half_life[0=重みなし], 旧CV5foldの値)
SETTINGS = [
    ("2015-01 / 重みなし (e02)", "2015-01-01", 0,   0.3856),
    ("2014-04 / 重みなし (e03)", "2014-04-01", 0,   0.3850),
    ("2014-04 / hl=180 (e12)",  "2014-04-01", 180, 0.3835),
    ("2014-04 / hl=365 (e09)",  "2014-04-01", 365, 0.3837),
]

df = F.build()
rows = []
for label, st, hl, old in SETTINGS:
    st = pd.Timestamp(st)
    per = {}
    for fl, vs, ve, drop in cv2.FOLDS:
        m_tr = (df.date >= st) & (df.date < vs) & df.sales.notna()
        m_va = (df.date >= vs) & (df.date <= ve)
        if drop:
            m_va = m_va & (~df.store_nbr.isin(drop))
        if m_tr.sum() < 200_000:   # fold B は 2016-08-15 までしか学習できない設定がある
            per[fl] = float("nan"); continue
        w = 0.5 ** ((vs - df.loc[m_tr, "date"]).dt.days.values / hl) if hl else None
        ds = lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"], weight=w, categorical_feature=F.CATS)
        mdl = lgb.train(PARAMS, ds, num_boost_round=900)  # e08 より iter 依存は極小
        p = np.clip(mdl.predict(df.loc[m_va, F.FEATURES]), 0, None)
        per[fl] = float(np.sqrt(np.mean((p - df.loc[m_va, "y"].values) ** 2)))
    rows.append(dict(label=label, old5=old, **per, new2=np.mean(list(per.values()))))
    print(f"{label:26s} A {per['A:2017-08前半']:.4f}  B {per['B:2016-08後半']:.4f}  平均 {rows[-1]['new2']:.4f}  (旧5fold {old:.4f})", flush=True)

r = pd.DataFrame(rows)
base = r[r.label.str.contains("e02")].iloc[0]
print("\n== e02 を基準にした差分（負が改善）==")
print(f"{'設定':26s} {'旧5fold差':>10s} {'新A差':>9s} {'新B差':>9s} {'新2fold差':>10s}  符号一致")
for _, x in r.iterrows():
    da, db = x["A:2017-08前半"] - base["A:2017-08前半"], x["B:2016-08後半"] - base["B:2016-08後半"]
    agree = "○" if (da < 0) == (db < 0) else "×"
    print(f"{x.label:26s} {x.old5-base.old5:+10.4f} {da:+9.4f} {db:+9.4f} {x.new2-base.new2:+10.4f}     {agree}")
print("\n== 旧5fold と 新2fold の順位相関 ==")
print("  spearman: %.3f" % r[["old5", "new2"]].corr(method="spearman").iloc[0, 1])
print("  旧CV best:", r.loc[r.old5.idxmin(), "label"], "/ 新CV best:", r.loc[r.new2.idxmin(), "label"])
r.to_csv(c.OUTPUT / "e14_cv2_recheck.csv", index=False)
