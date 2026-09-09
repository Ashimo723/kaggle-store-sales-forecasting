"""e18: 拡張特徴（94個）を新CVで評価。ベース = 56特徴 / 2014-04起点 / hl=365 = 0.3828。
効いた場合はグループ別 ablation で寄与を分解する。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, features2 as F2, cv2
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
TRAIN_START, HL = pd.Timestamp("2014-04-01"), 365

df = F2.build()
GROUPS = {
    "全94特徴": F2.FEATURES,
    "56特徴(ベース)": F.FEATURES,
    "+transactions のみ": F.FEATURES + [x for x in F2.NEW if x.startswith("tx_")],
    "+集約系列のみ": F.FEATURES + [x for x in F2.NEW if x.startswith("ag")],
    "+未来promoのみ": F.FEATURES + [x for x in F2.NEW if x.startswith("promo_")],
    "+状態量のみ": F.FEATURES + [x for x in F2.NEW if x.split("0")[0] in ("zero_rate", "rmed", "rmax", "rmin", "ewm") or x == "days_since_sale"],
    "+カレンダー2のみ": F.FEATURES + ["days_to_hol", "days_from_hol", "week_of_month", "is_month_end3", "eq_days"],
}
rows = []
for label, feats in GROUPS.items():
    per, t0 = {}, time.time()
    for fl, vs, ve, drop in cv2.FOLDS:
        m_tr = (df.date >= TRAIN_START) & (df.date < vs) & df.sales.notna()
        m_va = (df.date >= vs) & (df.date <= ve) & (~df.store_nbr.isin(drop) if drop else True)
        w = 0.5 ** ((vs - df.loc[m_tr, "date"]).dt.days.values / HL)
        mdl = lgb.train(P, lgb.Dataset(df.loc[m_tr, feats], df.loc[m_tr, "y"], weight=w,
                                       categorical_feature=F2.CATS), num_boost_round=900)
        p = np.clip(mdl.predict(df.loc[m_va, feats]), 0, None)
        per[fl] = float(np.sqrt(np.mean((p - df.loc[m_va, "y"].values) ** 2)))
        if label == "全94特徴" and fl.startswith("A"):
            imp_a = pd.Series(mdl.feature_importance("gain"), index=feats)
    rows.append(dict(label=label, n=len(feats), **per, mean=np.mean(list(per.values()))))
    print(f"{label:20s} ({len(feats):3d}特徴) A {per['A:2017-08前半']:.4f}  B {per['B:2016-08後半']:.4f}  平均 {rows[-1]['mean']:.4f}  {time.time()-t0:.0f}s", flush=True)

r = pd.DataFrame(rows); b = r[r.label.str.contains("ベース")].iloc[0]
print("\n== 56特徴ベース比（負が改善 / 採用は符号一致かつ |差|>=0.002）==")
for _, x in r.iterrows():
    da, db = x["A:2017-08前半"] - b["A:2017-08前半"], x["B:2016-08後半"] - b["B:2016-08後半"]
    tag = "★改善" if (da < 0 and db < 0 and abs(x["mean"] - b["mean"]) >= 0.002) else ("同方向だが小" if (da < 0) == (db < 0) else "符号不一致")
    print(f"{x.label:20s} A {da:+.4f}  B {db:+.4f}  平均 {x['mean']-b['mean']:+.4f}   {tag}")
print("\n== 全94特徴・fold A の重要度 top25 ==")
print((imp_a / imp_a.sum()).sort_values(ascending=False).head(25).round(4).to_string())
print("\n== 新規特徴の合計 gain 比: %.4f ==" % (imp_a[[x for x in F2.NEW if x in imp_a.index]].sum() / imp_a.sum()))
r.to_csv(c.OUTPUT / "e18_features2.csv", index=False)
