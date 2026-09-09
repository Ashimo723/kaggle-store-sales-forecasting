"""e36: family ごとに独立したモデルを学習する（公開ノート LB 0.37984 の構成）。

e34 ではゼロ率帯4分割が悪化したが、family 分割は未検証。
family(33) は store(54) より粒度が粗く、1 family あたり 54店舗 × 日数 のデータが残る。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv3
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
ST, SEEDS = pd.Timestamp("2015-01-01"), [42, 1, 2]
FEATS_NF = [f for f in F.FEATURES if f != "family"]      # family 別モデルでは family 列は定数
CATS_NF = [x for x in F.CATS if x != "family"]
df = F.build()

rec = {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    m_va = (df.date >= vs) & (df.date <= ve) & df.sales.notna()
    if m_va.sum() == 0: continue
    m_tr = (df.date >= ST) & (df.date < vs) & df.sales.notna()
    tr, va = df[m_tr], df[m_va]
    yv = va["y"].values

    ps = []
    for sd in SEEDS:
        mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                        lgb.Dataset(tr[F.FEATURES], tr["y"], categorical_feature=F.CATS), num_boost_round=500)
        ps.append(np.clip(mdl.predict(va[F.FEATURES]), 0, None))
    base = np.mean(ps, axis=0)

    pred = np.zeros(len(va))
    va_fam = va["family"].values
    for fam, g in tr.groupby("family", observed=True):
        mva = va_fam == fam
        if mva.sum() == 0: continue
        pb = []
        for sd in SEEDS:
            m2 = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                           lgb.Dataset(g[FEATS_NF], g["y"], categorical_feature=CATS_NF), num_boost_round=500)
            pb.append(np.clip(m2.predict(va[mva][FEATS_NF]), 0, None))
        pred[mva] = np.mean(pb, axis=0)
    row = {"base": float(np.sqrt(np.mean((base - yv) ** 2))),
           "byfam": float(np.sqrt(np.mean((pred - yv) ** 2)))}
    for w in [0.3, 0.5]:
        row[f"blend{w}"] = float(np.sqrt(np.mean(((1 - w) * base + w * pred - yv) ** 2)))
    rec[fname] = row
    print(f"{fname}  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()) + f"  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec).T
R.to_csv(c.OUTPUT / "e36_family.csv")
print(f"\n== {len(R)} fold 平均 ==")
print(R.mean().round(4).to_string())
print("\n== 対比較（基準 base、採用は |t|>=3）==")
for col in R.columns:
    if col == "base": continue
    d = (R[col] - R["base"]).dropna(); se = d.std() / np.sqrt(len(d)); t = d.mean() / se
    print(f"  {col:9s} 平均差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}% ({(d<0).sum()}/{len(d)})  t {t:+.2f}  {'★採用可' if t <= -3 else ''}")
