"""e33: 階層モデル（店舗合計の予測で水準を補正する）。

動機: 現行は 1782系列を**完全に独立**に予測しており、店舗レベルの整合制約が一切ない。
店舗合計は 54系列しかなく系列あたりのデータが厚いので、水準を高精度に推定できるはず
（e19: transactions の16日先予測 RMSE 0.0972 が実証）。

方式: 予測を作り直さず**水準だけ補正**する（既存資産を壊さない後処理）。
  1. 店舗×日の合計売上を fold 内で学習し16日先予測 → S
  2. 現行予測の店舗合計 T = Σ_f expm1(pred)
  3. 比 r = S / T を各行に r^alpha 倍で適用（alpha=0 は無補正、1 は完全一致）

判定: cv3 27fold 対比較、|t|>=3。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv3
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
PT = {**_P, "num_leaves": 63, "min_data_in_leaf": 40, "num_threads": 8}
ST, SEEDS = pd.Timestamp("2015-01-01"), [42, 1, 2]
ALPHAS = [0.25, 0.5, 0.75, 1.0]

df = F.build()

# ---- 店舗×日の合計売上パネル（原空間の合計を log1p）----
tot = (df[~df.is_test].groupby(["store_nbr", "date"], observed=True)["sales"].sum()
       .rename("tot").reset_index())
tot["store_nbr"] = tot.store_nbr.astype(int)
tot["ty"] = np.log1p(tot["tot"]).astype("float32")
# カレンダー/祝日/oil/promo を店舗×日単位で借りる
cols = ["dow", "day", "month", "year", "dayofyear", "weekofyear", "is_weekend", "is_payday",
        "days_since_payday", "days_to_payday", "is_nye", "hol_national", "hol_regional",
        "hol_local", "hol_any", "is_event", "is_workday", "dcoilwtico", "oil_ma28", "cluster"]
g = df.groupby(["store_nbr", "date"], observed=True)
side = g[cols].first().reset_index()
side["promo_store"] = g["onpromotion"].sum().values.astype("float32")
side["store_nbr"] = side.store_nbr.astype(int)
tot = side.merge(tot[["store_nbr", "date", "ty"]], on=["store_nbr", "date"], how="left")
tot = tot.sort_values(["store_nbr", "date"], ignore_index=True)
gt = tot.groupby("store_nbr", observed=True)["ty"]
TL = [16, 17, 21, 28, 35, 42, 364]
for l in TL:
    tot[f"tlag{l}"] = gt.shift(l).astype("float32")
base = gt.shift(16); gb = base.groupby(tot.store_nbr, observed=True)
for w in [7, 28, 112, 365]:
    tot[f"trm{w}"] = gb.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean()).astype("float32")
tot["tdow4"] = tot[["tlag21", "tlag28", "tlag35", "tlag42"]].mean(axis=1).astype("float32")
tot["cluster"] = tot["cluster"].astype("category")
TF = ([c_ for c_ in cols if c_ != "cluster"] + ["cluster", "promo_store"]
      + [f"tlag{l}" for l in TL] + [f"trm{w}" for w in [7, 28, 112, 365]] + ["tdow4"])

rec = {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    m_va = (df.date >= vs) & (df.date <= ve) & df.sales.notna()
    if m_va.sum() == 0: continue
    m_tr = (df.date >= ST) & (df.date < vs) & df.sales.notna()
    yv = df.loc[m_va, "y"].values

    ps = []
    for sd in SEEDS:
        mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                        lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"], categorical_feature=F.CATS),
                        num_boost_round=500)
        ps.append(np.clip(mdl.predict(df.loc[m_va, F.FEATURES]), 0, None))
    L = np.mean(ps, axis=0)

    # 店舗合計モデル（fold 内で学習）
    a_tr = (tot.date >= ST) & (tot.date < vs) & tot.ty.notna()
    a_va = (tot.date >= vs) & (tot.date <= ve)
    tm = lgb.train(PT, lgb.Dataset(tot.loc[a_tr, TF], tot.loc[a_tr, "ty"],
                                   categorical_feature=["cluster"]), num_boost_round=400)
    S_pred = np.expm1(np.clip(tm.predict(tot.loc[a_va, TF]), 0, None))
    S = pd.Series(S_pred, index=pd.MultiIndex.from_arrays(
        [tot.loc[a_va, "store_nbr"].values, tot.loc[a_va, "date"].values]))
    # 店舗合計モデル単体の精度（真値と比較）
    true_tot = df[m_va].groupby(["store_nbr", "date"], observed=True)["sales"].sum()
    idx = pd.MultiIndex.from_arrays([true_tot.index.get_level_values(0).astype(int),
                                     true_tot.index.get_level_values(1)])
    s_al = S.reindex(idx).values
    tot_rmsle = float(np.sqrt(np.nanmean((np.log1p(s_al) - np.log1p(true_tot.values)) ** 2)))

    va = df.loc[m_va, ["store_nbr", "date"]].copy()
    va["p"] = np.expm1(L)
    T = va.groupby(["store_nbr", "date"], observed=True)["p"].transform("sum").values
    key = pd.MultiIndex.from_arrays([va.store_nbr.astype(int).values, va.date.values])
    Sv = S.reindex(key).values
    r = np.where((T > 1e-6) & np.isfinite(Sv), Sv / np.maximum(T, 1e-6), 1.0)
    r = np.clip(r, 0.5, 2.0)   # 暴走防止

    row = {"L": float(np.sqrt(np.mean((L - yv) ** 2))), "tot_rmsle": tot_rmsle,
           "r_med": float(np.median(r))}
    for a in ALPHAS:
        q = np.log1p(np.expm1(L) * (r ** a))
        row[f"a{a}"] = float(np.sqrt(np.mean((np.clip(q, 0, None) - yv) ** 2)))
    rec[fname] = row
    print(f"{fname}  L {row['L']:.4f}  " + "  ".join(f"a{a} {row[f'a{a}']:.4f}" for a in ALPHAS)
          + f"  | 店舗合計モデル RMSLE {tot_rmsle:.4f}  r中央値 {row['r_med']:.3f}  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec).T
R.to_csv(c.OUTPUT / "e33_hier.csv")
print(f"\n== {len(R)} fold 平均 ==")
print(R.mean().round(4).to_string())
print("\n== 対比較（基準 L、採用は |t|>=3）==")
for a in ALPHAS:
    d = (R[f"a{a}"] - R["L"]).dropna()
    se = d.std() / np.sqrt(len(d)); t = d.mean() / se
    print(f"  alpha={a:.2f}  平均差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}% ({(d<0).sum()}/{len(d)})  t {t:+.2f}  {'★採用可' if t <= -3 else ''}")
