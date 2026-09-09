"""e23: fold の代表性診断（ユーザー提案「店舗セグメントごとの層化」の時系列版）。

時系列 CV では日付で切るのが唯一正しく、ランダム層化分割は使えない。
代わりに **各 fold の誤差をセグメント別に分解し、fold 間で構造が一致するか** を見る。
一致しなければ 2fold での判定が偏っている証拠になり、fold を増やす/重み付ける根拠になる。

セグメント: store type(A-E) / cluster(17) / 店舗規模四分位 / family / 系列のゼロ率帯
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv2
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
TRAIN_START, HL = pd.Timestamp("2014-04-01"), 365

df = F.build()
# 追加の検証 fold（代表性を見るため 2fold 以外も測る）
EXTRA = [("C:2017-07前半", pd.Timestamp("2017-07-15"), pd.Timestamp("2017-07-30"), set()),
         ("D:2017-06後半", pd.Timestamp("2017-06-29"), pd.Timestamp("2017-07-14"), set()),
         ("E:2015-08後半", pd.Timestamp("2015-08-16"), pd.Timestamp("2015-08-31"), set())]
ALL_FOLDS = list(cv2.FOLDS) + EXTRA

preds = {}
for fl, vs, ve, drop in ALL_FOLDS:
    m_tr = (df.date >= TRAIN_START) & (df.date < vs) & df.sales.notna()
    m_va = (df.date >= vs) & (df.date <= ve) & (~df.store_nbr.isin(drop) if drop else True)
    w = 0.5 ** ((vs - df.loc[m_tr, "date"]).dt.days.values / HL)
    ps = []
    for sd in (42, 1):
        mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                        lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"], weight=w,
                                    categorical_feature=F.CATS), num_boost_round=900)
        ps.append(np.clip(mdl.predict(df.loc[m_va, F.FEATURES]), 0, None))
    v = df.loc[m_va, ["date", "store_nbr", "family", "type", "cluster", "y"]].copy()
    v["pred"] = np.mean(ps, axis=0)
    v["se"] = (v.pred - v.y) ** 2
    preds[fl] = v
    print(f"{fl}  RMSLE {np.sqrt(v.se.mean()):.4f}  n {len(v):,}", flush=True)

# 店舗規模四分位（train 期間の平均売上で定義）
lvl = df[(df.date >= "2017-01-01") & (df.date < "2017-07-31")].groupby("store_nbr", observed=True)["y"].mean()
size_q = pd.qcut(lvl, 4, labels=["小", "中小", "中大", "大"])
# 系列のゼロ率帯
zr = df[(df.date >= "2017-01-01") & (df.date < "2017-07-31")].groupby(["store_nbr", "family"], observed=True)["y"].apply(lambda s: (s <= 1e-9).mean())
zr_band = pd.cut(zr, [-.01, .05, .3, .7, 1.01], labels=["密", "中", "疎", "ほぼ0"])

print("\n" + "=" * 78)
for seg_name, getter in [
    ("store type", lambda v: v["type"].astype(str)),
    ("cluster", lambda v: v["cluster"].astype(str)),
    ("店舗規模", lambda v: v["store_nbr"].map(size_q).astype(str)),
    ("ゼロ率帯", lambda v: pd.MultiIndex.from_arrays([v.store_nbr, v.family]).map(zr_band).astype(str)),
]:
    tab = {}
    for fl, v in preds.items():
        g = v.assign(seg=getter(v)).groupby("seg", observed=True)
        tab[fl] = np.sqrt(g.se.mean())
    t = pd.DataFrame(tab)
    print(f"\n== セグメント別 RMSLE: {seg_name} ==")
    print(t.round(4).to_string())
    ab = t[["A:2017-08前半", "B:2016-08後半"]].dropna()
    print(f"  fold A-B のセグメント別 RMSLE 相関: pearson {ab.corr().iloc[0,1]:.3f} / spearman {ab.corr(method='spearman').iloc[0,1]:.3f}")

print("\n== fold 別 総合 RMSLE と SSE 構成（上位セグメントの寄与）==")
for fl, v in preds.items():
    top = v.groupby("type", observed=True).se.sum()
    print(f"  {fl}: RMSLE {np.sqrt(v.se.mean()):.4f}   type別SSE比 " +
          " ".join(f"{k}:{x:.2f}" for k, x in (top / top.sum()).items()))
