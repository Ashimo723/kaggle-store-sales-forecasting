"""e34: セグメント別モデル。系列をゼロ率帯で分け、帯ごとに独立した LightGBM を学習する。

動機: e24 でゼロ率帯ごとに RMSLE が 0.25〜0.63 と 2.5倍違う。
      BOOKS(96.9%ゼロ) と GROCERY I(0.4%ゼロ) を同じ木・同じ損失で扱う無理を解消できるか。
懸念: LightGBM は既に family/store をカテゴリ分岐に使っており、木の上位で系列タイプを分けている可能性。
      その場合 e33（階層構造は暗黙に学習済み）と同じ結末になる。データ量が 1/N になる副作用もある。

アーム:
  base    : 単一モデル（現行）
  seg4    : ゼロ率帯4つ（密/中/疎/ほぼ0）で別モデル
  seg2    : 密 と それ以外 の2分割（データ量の損失を抑えた版）
判定: cv3 27fold 対比較、|t|>=3。
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


def zero_band(train_df, cut4=True):
    """学習期間の直近1年のゼロ率で系列を帯に分ける（valid の情報は使わない）"""
    recent = train_df[train_df.date >= train_df.date.max() - pd.Timedelta(days=365)]
    zr = recent.groupby(["store_nbr", "family"], observed=True)["y"].apply(lambda s: (s <= 1e-9).mean())
    if cut4:
        return pd.cut(zr, [-.01, .05, .3, .7, 1.01], labels=[0, 1, 2, 3]).astype(float)
    return pd.cut(zr, [-.01, .05, 1.01], labels=[0, 1]).astype(float)


rec = {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    m_va = (df.date >= vs) & (df.date <= ve) & df.sales.notna()
    if m_va.sum() == 0: continue
    m_tr = (df.date >= ST) & (df.date < vs) & df.sales.notna()
    tr, va = df[m_tr], df[m_va]
    yv = va["y"].values
    row = {}

    # base: 単一モデル
    ps = []
    for sd in SEEDS:
        mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                        lgb.Dataset(tr[F.FEATURES], tr["y"], categorical_feature=F.CATS), num_boost_round=500)
        ps.append(np.clip(mdl.predict(va[F.FEATURES]), 0, None))
    row["base"] = float(np.sqrt(np.mean((np.mean(ps, axis=0) - yv) ** 2)))

    for tag, cut4 in [("seg4", True), ("seg2", False)]:
        band = zero_band(tr, cut4)
        tr_b = pd.MultiIndex.from_arrays([tr.store_nbr, tr.family]).map(band).values
        va_b = pd.MultiIndex.from_arrays([va.store_nbr, va.family]).map(band).values
        pred = np.zeros(len(va))
        for b in np.unique(tr_b[~pd.isna(tr_b)]):
            mtr = tr_b == b; mva = va_b == b
            if mva.sum() == 0: continue
            if mtr.sum() < 20000:      # データが少なすぎる帯は base 予測で埋める
                pred[mva] = np.mean(ps, axis=0)[mva]; continue
            pb = []
            for sd in SEEDS:
                m2 = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                               lgb.Dataset(tr[mtr][F.FEATURES], tr[mtr]["y"], categorical_feature=F.CATS),
                               num_boost_round=500)
                pb.append(np.clip(m2.predict(va[mva][F.FEATURES]), 0, None))
            pred[mva] = np.mean(pb, axis=0)
        miss = pd.isna(va_b)
        if miss.any(): pred[miss] = np.mean(ps, axis=0)[miss]
        row[tag] = float(np.sqrt(np.mean((pred - yv) ** 2)))
    rec[fname] = row
    print(f"{fname}  base {row['base']:.4f}  seg4 {row['seg4']:.4f}  seg2 {row['seg2']:.4f}  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec).T
R.to_csv(c.OUTPUT / "e34_segment.csv")
print(f"\n== {len(R)} fold 平均 ==")
print(R.mean().round(4).to_string())
print("\n== 対比較（基準 base、採用は |t|>=3）==")
for col in ["seg4", "seg2"]:
    d = (R[col] - R["base"]).dropna()
    se = d.std() / np.sqrt(len(d)); t = d.mean() / se
    print(f"  {col}  平均差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}% ({(d<0).sum()}/{len(d)})  t {t:+.2f}  {'★採用可' if t <= -3 else ''}")
