"""e32: 移動平均どうしの「差（= log 比）」を明示的に与える。

動機: LightGBM は軸平行分割しかできないので rmean7 - rmean28 のような特徴を自力で作れない。
既存特徴は絶対水準（log1p 値）ばかりで系列の規模に依存する。差を取れば規模非依存の「勢い」になる。
e18 の特徴追加（情報を足す）とは性質が違い、**同じ情報を使いやすい形に変換する**試み。

判定: cv3 27fold の対比較、|t|>=3。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv3
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
ST, SEEDS = pd.Timestamp("2015-01-01"), [42, 1, 2]

df = F.build()
# --- 比（log 空間の差）と規模非依存量 ---
df["mom_7_28"] = (df.rmean7 - df.rmean28).astype("float32")       # 短期 vs 中期モメンタム
df["mom_7_365"] = (df.rmean7 - df.rmean365).astype("float32")     # 直近 vs 年間平均
df["mom_28_112"] = (df.rmean28 - df.rmean112).astype("float32")   # 中期 vs 長期
df["dow_effect"] = (df.dow4_mean - df.rmean28).astype("float32")  # その曜日の効き（系列別）
df["last_dev"] = (df.lag16 - df.rmean7).astype("float32")         # 直近1日の異常度
df["cv28"] = (df.rstd28 / (df.rmean28.abs() + 0.5)).astype("float32")   # 変動係数
df["yoy_lvl"] = (df.rmean7 - df.lag364).astype("float32")         # 前年同日との水準差
# --- 年次季節性のフーリエ項 ---
doy = df["dayofyear"].values / 365.25
for k in [1, 2, 3]:
    df[f"sin{k}"] = np.sin(2 * np.pi * k * doy).astype("float32")
    df[f"cos{k}"] = np.cos(2 * np.pi * k * doy).astype("float32")

RATIO = ["mom_7_28", "mom_7_365", "mom_28_112", "dow_effect", "last_dev", "cv28", "yoy_lvl"]
FOURIER = [f"{f}{k}" for k in [1, 2, 3] for f in ["sin", "cos"]]
ARMS = {
    "base(56)": F.FEATURES,
    "+比(63)": F.FEATURES + RATIO,
    "+フーリエ(62)": F.FEATURES + FOURIER,
    "+比+フーリエ(69)": F.FEATURES + RATIO + FOURIER,
}
rec = {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    m_va = (df.date >= vs) & (df.date <= ve) & df.sales.notna()
    if m_va.sum() == 0: continue
    m_tr = (df.date >= ST) & (df.date < vs) & df.sales.notna()
    yv = df.loc[m_va, "y"].values
    row = {}
    for arm, feats in ARMS.items():
        ps = []
        for sd in SEEDS:
            mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                            lgb.Dataset(df.loc[m_tr, feats], df.loc[m_tr, "y"], categorical_feature=F.CATS),
                            num_boost_round=500)
            ps.append(np.clip(mdl.predict(df.loc[m_va, feats]), 0, None))
        row[arm] = float(np.sqrt(np.mean((np.mean(ps, axis=0) - yv) ** 2)))
        if arm == "+比+フーリエ(69)" and fname == cv3.FOLDS[-1][0]:
            imp = pd.Series(mdl.feature_importance("gain"), index=feats)
    rec[fname] = row
    print(f"{fname}  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()) + f"  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec).T
R.to_csv(c.OUTPUT / "e32_ratio.csv")
print(f"\n== {len(R)} fold 平均 ==")
print(R.mean().round(4).to_string())
print("\n== 対比較（基準 base(56)、採用は |t|>=3）==")
for col in R.columns:
    if col == "base(56)": continue
    d = (R[col] - R["base(56)"]).dropna()
    se = d.std() / np.sqrt(len(d)); t = d.mean() / se
    print(f"  {col:18s} 平均差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}% ({(d<0).sum()}/{len(d)})  t {t:+.2f}  {'★採用可' if t <= -3 else ''}")
try:
    print("\n== 新規特徴の gain 比（最終 fold）==")
    print((imp / imp.sum())[RATIO + FOURIER].sort_values(ascending=False).round(4).to_string())
    print("（参考 rmean7: %.4f）" % (imp["rmean7"] / imp.sum()))
except Exception as e:
    print(e)
