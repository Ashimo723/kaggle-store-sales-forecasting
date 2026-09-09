"""e39: (1) store別モデル と (3) family別+祝日ダミー を同時に検証する。

e36 の発見:「同じ特徴・同じアルゴリズムで**学習単位だけ変える**」と直交性と単体性能を両立できる。
  → (1) 別の切り口（store 54分割）でも同じ原理が働くはず。family別と互いに直交すれば3本ブレンドできる
  → (3) 祝日ダミーは単一モデルでは年3-5サンプルで学習不能だった（e35, gain 0.00004）。
        family別なら「その family 固有の祝日反応」になり学習条件が変わる（GRAVEYARD g15 の再挑戦条件）

各 fold で4種の予測を作り npz に保存 → 後から任意の比率を再評価できる。
対比較なので seed は 42 の1本に統一（同一 seed 同士の比較なら seed ノイズは相殺される）。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv3, holidays2
from config import PARAMS as _P
P = {**_P, "num_threads": 8, "seed": 42, "bagging_seed": 42, "feature_fraction_seed": 42}
ST, NR = pd.Timestamp("2015-01-01"), 500

df = F.build()
_, nat14, loc, reg = holidays2.build(selected=[
    "navidad", "terremoto", "independencia", "primer dia ano", "futbol", "carnaval",
    "dia la madre", "primer grito independencia", "dia difuntos", "batalla",
    "viernes santo", "dia trabajo", "black friday", "cyber monday"])
N14 = [x for x in nat14.columns if x != "date"]
df = df.merge(nat14, on="date", how="left")
df[N14] = df[N14].fillna(0).astype("int8")
for col in ["family", "city", "state", "type", "store_nbr", "cluster"]:
    df[col] = df[col].astype("category")

FEATS_NF = [f for f in F.FEATURES if f != "family"]           # family別: family は定数
CATS_NF = [x for x in F.CATS if x != "family"]
FEATS_NS = [f for f in F.FEATURES if f != "store_nbr"]        # store別: store と city/state/type/cluster が定数
FEATS_NS = [f for f in FEATS_NS if f not in ("city", "state", "type", "cluster")]
CATS_NS = [x for x in F.CATS if x in FEATS_NS]

rec, store_pred = {}, {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    m_va = (df.date >= vs) & (df.date <= ve) & df.sales.notna()
    if m_va.sum() == 0: continue
    m_tr = (df.date >= ST) & (df.date < vs) & df.sales.notna()
    tr, va = df[m_tr], df[m_va]
    yv = va["y"].values

    base = np.clip(lgb.train(P, lgb.Dataset(tr[F.FEATURES], tr["y"], categorical_feature=F.CATS),
                             num_boost_round=NR).predict(va[F.FEATURES]), 0, None)

    fam = np.zeros(len(va)); famh = np.zeros(len(va)); vf = va["family"].values
    for f_, g in tr.groupby("family", observed=True):
        m = vf == f_
        if m.sum() == 0: continue
        fam[m] = np.clip(lgb.train(P, lgb.Dataset(g[FEATS_NF], g["y"], categorical_feature=CATS_NF),
                                   num_boost_round=NR).predict(va[m][FEATS_NF]), 0, None)
        fh = FEATS_NF + N14
        famh[m] = np.clip(lgb.train(P, lgb.Dataset(g[fh], g["y"], categorical_feature=CATS_NF),
                                    num_boost_round=NR).predict(va[m][fh]), 0, None)

    sto = np.zeros(len(va)); vs_ = va["store_nbr"].values
    for s_, g in tr.groupby("store_nbr", observed=True):
        m = vs_ == s_
        if m.sum() == 0: continue
        sto[m] = np.clip(lgb.train(P, lgb.Dataset(g[FEATS_NS], g["y"], categorical_feature=CATS_NS),
                                   num_boost_round=NR).predict(va[m][FEATS_NS]), 0, None)

    np.savez_compressed(c.OUTPUT / f"e39_pred_{fname}.npz", y=yv, base=base, fam=fam, famh=famh, sto=sto)
    r = lambda p: float(np.sqrt(np.mean((p - yv) ** 2)))
    rec[fname] = {"base": r(base), "fam": r(fam), "famh": r(famh), "sto": r(sto),
                  "b+fam.3": r(.7*base+.3*fam), "b+famh.3": r(.7*base+.3*famh),
                  "b+sto.3": r(.7*base+.3*sto), "b+sto.2": r(.8*base+.2*sto),
                  "3way": r(.55*base+.25*fam+.20*sto), "3way_h": r(.55*base+.25*famh+.20*sto)}
    print(f"{fname}  base {rec[fname]['base']:.4f}  fam {rec[fname]['fam']:.4f}  famh {rec[fname]['famh']:.4f}  "
          f"sto {rec[fname]['sto']:.4f}  | 3way {rec[fname]['3way']:.4f}  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec).T
R.to_csv(c.OUTPUT / "e39_store_holiday.csv")
print(f"\n== {len(R)} fold 平均 ==")
print(R.mean().round(4).to_string())
NY = [i for i in R.index if i == "2016-12-26"]
for label, S in [("全fold", R), ("年末年始を除く", R.drop(index=NY)), ("2017年のみ", R[[str(i).startswith("2017") for i in R.index]])]:
    print(f"\n== 対比較 基準base / {label} n={len(S)} ==")
    for col in R.columns:
        if col == "base": continue
        d = (S[col] - S["base"]).dropna(); se = d.std()/np.sqrt(len(d)); t = d.mean()/se
        ok = "★" if (100*(d<0).mean() >= 85 and d.median() <= d.mean()) or t <= -3 else ""
        print(f"  {col:10s} 平均差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}%  t {t:+.2f} {ok}")
