"""e48: XGBoost 脚を追加してアンサンブルする（ユーザー提案）。

e31 の教訓: 異質な脚（Ridge/Croston）は**単体性能が base から離れすぎ**て全滅した。
XGBoost / CatBoost は同じ GBDT で単体性能が近いはずなので、その制約に引っかからない。
direct 方式なので再帰予測のような暴走リスクもない（g17 の失敗を繰り返さない）。

判定は 2026-09-09 の新基準: 勝率>=85% かつ 中央値<=平均差 かつ **最悪 fold の悪化 <= +0.002**
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv3
from config import PARAMS as _P
PL = {**_P, "num_threads": 8, "seed": 42, "bagging_seed": 42, "feature_fraction_seed": 42}
ST, NR = pd.Timestamp("2015-01-01"), 500

df = F.build()
# XGBoost 用: category dtype をそのまま使う（enable_categorical）
XP = dict(max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
          reg_lambda=1.0, tree_method="hist", enable_categorical=True, max_cat_to_onehot=1,
          n_estimators=NR, n_jobs=8, random_state=42, verbosity=0)

rec = {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    m_va = (df.date >= vs) & (df.date <= ve) & df.sales.notna()
    if m_va.sum() == 0: continue
    m_tr = (df.date >= ST) & (df.date < vs) & df.sales.notna()
    yv = df.loc[m_va, "y"].values
    row = {}

    # (1) LightGBM 単体（対照）
    lm = lgb.train(PL, lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"],
                                   categorical_feature=F.CATS), num_boost_round=NR)
    p_lgb = np.clip(lm.predict(df.loc[m_va, F.FEATURES]), 0, None)
    row["lgb"] = float(np.sqrt(np.mean((p_lgb - yv) ** 2)))

    # (2) XGBoost
    xm = xgb.XGBRegressor(**XP).fit(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"], verbose=False)
    p_xgb = np.clip(xm.predict(df.loc[m_va, F.FEATURES]), 0, None)
    row["xgb"] = float(np.sqrt(np.mean((p_xgb - yv) ** 2)))

    # 残差相関（直交性の指標）
    rl = p_lgb - yv
    row["corr_xgb"] = float(np.corrcoef(rl, p_xgb - yv)[0, 1])

    # (4) 現 best 構成（e44 = 4本ブレンド）を npz から取得し、その上に積む
    a = np.load(c.OUTPUT / f"e39_pred_{fname}.npz"); b = np.load(c.OUTPUT / f"e40_pred_{fname}.npz")
    g = np.load(c.OUTPUT / f"e42_gseas_{fname}.npz")
    best4 = 0.4*a["base"] + 0.2*a["fam"] + 0.2*b["famgrp"] + 0.2*g["p"]
    row["best4"] = float(np.sqrt(np.mean((best4 - yv) ** 2)))
    for w in [0.10, 0.15, 0.20, 0.25, 0.35]:
        row[f"best4+xgb{w}"] = float(np.sqrt(np.mean(((1-w)*best4 + w*p_xgb - yv) ** 2)))
    np.savez_compressed(c.OUTPUT / f"e48_pred_{fname}.npz", xgb=p_xgb)
    rec[fname] = row
    print(f"{fname}  lgb {row['lgb']:.4f}  xgb {row['xgb']:.4f}  | 残差corr {row['corr_xgb']:.3f}  "
          f"best4 {row['best4']:.4f}  +xgb.15 {row['best4+xgb0.15']:.4f}  +xgb.25 {row['best4+xgb0.25']:.4f}  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec).T
R.to_csv(c.OUTPUT / "e48_xgb_cat.csv")
print(f"\n== {len(R)} fold 平均 ==")
print(R.mean().round(4).to_string())
NY = ["2016-12-26"]
for label, S in [("年末年始を除く", R.drop(index=[i for i in R.index if i in NY])),
                 ("2017年のみ", R[[str(i).startswith("2017") for i in R.index]])]:
    print(f"\n== 基準 best4 / {label} n={len(S)} ==")
    for col in R.columns:
        if not col.startswith("best4+"): continue
        d = (S[col] - S["best4"]).dropna(); se = d.std()/np.sqrt(len(d)); t = d.mean()/se
        win, med, worst = 100*(d<0).mean(), d.median(), d.max()
        ok = "★採用可" if (win >= 85 and med <= d.mean() and worst <= 0.002) else ""
        print(f"  {col:18s} 差 {d.mean():+.4f}  中央値 {med:+.4f}  勝率 {win:3.0f}%  最悪 {worst:+.4f}  t {t:+.2f} {ok}")
