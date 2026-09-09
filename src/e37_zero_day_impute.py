"""e37: 「店が閉まっていた日」を学習から外す / 補間する（公開ノート LB 0.37984 の知見）。

現行は 1/1（ほぼ全店休業、e01b で全店合計が通常の1/100以下と確認）や
店舗単位の休業日（e13: store 18/25 は16日間まるごと 0）を **売上0のサンプルとして学習に入れている**。
これは「店が閉まっていた」を「売れなかった」として学習させることになり、系列の水準を引き下げる。

アーム:
  base        : 現行（全部そのまま学習）
  drop_nye    : 1/1 の行を学習から除外
  drop_closed : 店舗×日の売上合計が 0 の行（＝休業日）を学習から除外
  drop_both   : 両方
  interp      : 上記の行を NaN 扱いにせず、系列ごとに前後の線形補間で置き換えて学習
評価側の valid は一切変更しない（同じ土俵で比較する）。
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

# 店舗×日の売上合計が 0 = その日その店は閉まっていた
store_day = df[~df.is_test].groupby(["store_nbr", "date"], observed=True)["sales"].transform("sum")
df["closed"] = (store_day.reindex(df.index).fillna(1) <= 0).astype("int8")
df["is_nye_day"] = ((df.date.dt.month == 1) & (df.date.dt.day == 1)).astype("int8")
print(f"休業行 {df.closed.sum():,} / 1-1 行 {df.is_nye_day.sum():,} / 全 {len(df):,}")

# interp 用: 休業/1-1 の y を系列ごとに線形補間した列
y_int = df["y"].copy()
mask_bad = (df.closed == 1) | (df.is_nye_day == 1)
y_int[mask_bad] = np.nan
df["y_interp"] = (y_int.groupby([df.store_nbr, df.family], observed=True)
                  .transform(lambda s: s.interpolate(limit_direction="both"))).astype("float32")

ARMS = ["base", "drop_nye", "drop_closed", "drop_both", "interp"]
rec = {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    m_va = (df.date >= vs) & (df.date <= ve) & df.sales.notna()
    if m_va.sum() == 0: continue
    m_tr0 = (df.date >= ST) & (df.date < vs) & df.sales.notna()
    yv = df.loc[m_va, "y"].values
    row = {}
    for arm in ARMS:
        m_tr, tgt = m_tr0, "y"
        if arm == "drop_nye":     m_tr = m_tr0 & (df.is_nye_day == 0)
        elif arm == "drop_closed": m_tr = m_tr0 & (df.closed == 0)
        elif arm == "drop_both":   m_tr = m_tr0 & (df.closed == 0) & (df.is_nye_day == 0)
        elif arm == "interp":      tgt = "y_interp"
        ps = []
        for sd in SEEDS:
            mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                            lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, tgt],
                                        categorical_feature=F.CATS), num_boost_round=500)
            ps.append(np.clip(mdl.predict(df.loc[m_va, F.FEATURES]), 0, None))
        row[arm] = float(np.sqrt(np.mean((np.mean(ps, axis=0) - yv) ** 2)))
    rec[fname] = row
    print(f"{fname}  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items()) + f"  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec).T
R.to_csv(c.OUTPUT / "e37_zeroday.csv")
print(f"\n== {len(R)} fold 平均 ==")
print(R.mean().round(4).to_string())
print("\n== 対比較（基準 base、採用は |t|>=3）==")
for col in R.columns:
    if col == "base": continue
    d = (R[col] - R["base"]).dropna(); se = d.std() / np.sqrt(len(d)); t = d.mean() / se
    print(f"  {col:12s} 平均差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}% ({(d<0).sum()}/{len(d)})  t {t:+.2f}  {'★採用可' if t <= -3 else ''}")
