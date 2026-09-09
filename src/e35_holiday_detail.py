"""e35: 祝日を種類ごとの個別ダミーに展開する（公開ノート LB 0.37984 からの知見）。

我々は hol_national/regional/local の 0/1 フラグに集約していた。祝日は種類ごとに効果が正反対
（クリスマスは売上急増、元日はほぼ全店休業）なので、これは明確な情報損失。
e18 の「特徴追加」とは違い、**捨てていた情報の復元**にあたる。

アーム:
  base        : 現行56特徴
  +nat7       : 公開ノートが選んだ全国祝日7種のダミーを追加
  +nat14      : 正規化後の全国祝日 全14種（black friday / cyber monday を含む）
  +nat14+loc  : 全国14種 + 地方/地域ダミー
  detail_only : 集約フラグ(hol_national/regional/local/hol_any)を外し詳細ダミーだけにする
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv3, holidays2
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
ST, SEEDS = pd.Timestamp("2015-01-01"), [42, 1, 2]

df = F.build()
work, nat7, loc, reg = holidays2.build()
_, nat14, _, _ = holidays2.build(selected=[
    "navidad", "terremoto", "independencia", "primer dia ano", "futbol", "carnaval",
    "dia la madre", "primer grito independencia", "dia difuntos", "batalla",
    "viernes santo", "dia trabajo", "black friday", "cyber monday"])

N7 = [x for x in nat7.columns if x != "date"]
N14 = [x for x in nat14.columns if x != "date"]
df = df.merge(nat14, on="date", how="left")
LOCC = [x for x in loc.columns if x.startswith("loc_")]
df = df.merge(loc, on=["date", "city"], how="left")
df = df.merge(reg, on=["date", "state"], how="left")
fill = N14 + LOCC + ["reg_prov"]
df[fill] = df[fill].fillna(0).astype("int8")
# merge でカテゴリ型が object に落ちるので復元する
for col in ["family", "city", "state", "type", "store_nbr", "cluster"]:
    df[col] = df[col].astype("category")

AGG = ["hol_national", "hol_regional", "hol_local", "hol_any"]
ARMS = {
    "base(56)": F.FEATURES,
    "+nat7": F.FEATURES + N7,
    "+nat14": F.FEATURES + N14,
    "+nat14+loc": F.FEATURES + N14 + LOCC + ["reg_prov"],
    "detail_only": [f for f in F.FEATURES if f not in AGG] + N14 + LOCC + ["reg_prov"],
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
                            lgb.Dataset(df.loc[m_tr, feats], df.loc[m_tr, "y"],
                                        categorical_feature=[x for x in F.CATS if x in feats]),
                            num_boost_round=500)
            ps.append(np.clip(mdl.predict(df.loc[m_va, feats]), 0, None))
        row[arm] = float(np.sqrt(np.mean((np.mean(ps, axis=0) - yv) ** 2)))
        if arm == "+nat14" and fname == cv3.FOLDS[-1][0]:
            imp = pd.Series(mdl.feature_importance("gain"), index=feats)
    rec[fname] = row
    print(f"{fname}  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()) + f"  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec).T
R.to_csv(c.OUTPUT / "e35_holiday.csv")
print(f"\n== {len(R)} fold 平均 ==")
print(R.mean().round(4).to_string())
print("\n== 対比較（基準 base(56)、採用は |t|>=3）==")
for col in R.columns:
    if col == "base(56)": continue
    d = (R[col] - R["base(56)"]).dropna()
    se = d.std() / np.sqrt(len(d)); t = d.mean() / se
    print(f"  {col:14s} 平均差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}% ({(d<0).sum()}/{len(d)})  t {t:+.2f}  {'★採用可' if t <= -3 else ''}")
try:
    print("\n== 祝日ダミーの gain 比（最終 fold, +nat14）==")
    print((imp / imp.sum())[N14].sort_values(ascending=False).round(5).to_string())
except Exception as e:
    print(e)
