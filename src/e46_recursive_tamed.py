"""e46: 再帰予測の暴走を抑える。

e45: 中央値 -0.0043 と大きく改善するが、少数 fold で発散（2017-04-17 で +0.085、z21 では +0.27）。
対策3つを同時に測る:
  (1) クリップ: 各ステップで予測を「その系列の過去実績レンジ」に制限し発散を根本で止める
  (2) ブレンド: direct / best4 と混ぜて暴走を薄める
  (3) h ハイブリッド: h<=k は再帰、h>k は direct
暴走 fold の診断（どの系列がどれだけ発散したか）も同時に出す。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, cv3, recursive as R
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
ST, SEEDS, NR = pd.Timestamp("2015-01-01"), [42, 1, 2], 500

df = R.build()
keys = pd.MultiIndex.from_tuples(
    df[["store_nbr", "family"]].drop_duplicates().sort_values(["store_nbr", "family"]).apply(tuple, axis=1).tolist(),
    names=["store_nbr", "family"])


def rec_predict(models, origin, clip_q=None):
    """clip_q: 直近91日の分位でクリップ（None ならクリップなし）"""
    piv = df.pivot_table(index=["store_nbr", "family"], columns="date", values="y",
                         aggfunc="mean", observed=True).reindex(keys)
    piv = piv.reindex(columns=pd.date_range(piv.columns.min(), piv.columns.max()))
    M = piv.values.astype("float32"); dates = list(piv.columns)
    d2p = {d: i for i, d in enumerate(dates)}; opos = d2p[origin]
    M[:, opos + 1:] = np.nan
    if clip_q is not None:
        w = M[:, max(0, opos + 1 - 91):opos + 1]
        hi = np.nanquantile(np.where(np.isnan(w), np.nan, w), clip_q, axis=1)
        hi = np.nan_to_num(hi, nan=np.inf)
    side = df.set_index(["store_nbr", "family", "date"])
    out = {}
    for h in range(1, 17):
        d = origin + pd.Timedelta(days=h)
        if d not in d2p: break
        pos = d2p[d]
        rows = side.loc[(slice(None), slice(None), d), :].reset_index().set_index(["store_nbr", "family"]).reindex(keys).reset_index()
        X = rows[R.STATIC + R.CAL].copy()
        for col in R.STATIC:
            X[col] = X[col].astype(df[col].dtype)
        for k, v in R._lag_block(M, pos).items():
            X[k] = v
        p = np.mean([np.clip(m.predict(X[R.FEATURES]), 0, None) for m in models], axis=0)
        if clip_q is not None:
            p = np.minimum(p, hi)          # 過去実績レンジを超えさせない
        M[:, pos] = p.astype("float32"); out[d] = p
    return out


rec = {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    origin = vs - pd.Timedelta(days=1)
    m_tr = (df.date >= ST) & (df.date <= origin) & df.sales.notna()
    if m_tr.sum() < 100000: continue
    models = [lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                        lgb.Dataset(df.loc[m_tr, R.FEATURES], df.loc[m_tr, "y"],
                                    categorical_feature=R.CATS), num_boost_round=NR) for sd in SEEDS]
    tv = df[(df.date >= vs) & (df.date <= ve) & df.sales.notna()][["store_nbr", "family", "date", "y"]]
    row = {}
    P_ = {}
    for tag, q_ in [("rec", None), ("rec_c99", 0.99), ("rec_c95", 0.95)]:
        pr = rec_predict(models, origin, clip_q=q_)
        PR = pd.concat([pd.DataFrame({"store_nbr": keys.get_level_values(0), "family": keys.get_level_values(1),
                                      "date": d, "p": p}) for d, p in pr.items()], ignore_index=True)
        mg = tv.merge(PR, on=["store_nbr", "family", "date"], how="left").sort_values(["store_nbr", "family", "date"])
        P_[tag] = mg
        row[tag] = float(np.sqrt(np.nanmean((mg.p - mg.y) ** 2)))
    a = np.load(c.OUTPUT / f"e39_pred_{fname}.npz"); b = np.load(c.OUTPUT / f"e40_pred_{fname}.npz")
    g = np.load(c.OUTPUT / f"e42_gseas_{fname}.npz")
    key_df = df[(df.date >= vs) & (df.date <= ve) & df.sales.notna()][["store_nbr", "family", "date"]].reset_index(drop=True)
    best4 = 0.4*a["base"] + 0.2*a["fam"] + 0.2*b["famgrp"] + 0.2*g["p"]
    ref = key_df.assign(best4=best4, y=a["y"]).sort_values(["store_nbr", "family", "date"])
    row["best4"] = float(np.sqrt(np.mean((ref.best4 - ref.y) ** 2)))
    for tag in ["rec", "rec_c95"]:
        pr = P_[tag].p.values
        for w in [0.2, 0.3, 0.4]:
            row[f"best4+{tag}{w}"] = float(np.sqrt(np.nanmean(((1-w)*ref.best4.values + w*pr - ref.y.values) ** 2)))
    rec[fname] = row
    print(f"{fname}  best4 {row['best4']:.4f}  rec {row['rec']:.4f}  c99 {row['rec_c99']:.4f}  c95 {row['rec_c95']:.4f}  "
          f"| +rec.3 {row['best4+rec0.3']:.4f}  +c95.3 {row['best4+rec_c950.3']:.4f}  {time.time()-t0:.0f}s", flush=True)

Rd = pd.DataFrame(rec).T
Rd.to_csv(c.OUTPUT / "e46_tamed.csv")
print(f"\n== {len(Rd)} fold 平均 ==")
print(Rd.mean().round(4).to_string())
NY = ["2016-12-26"]
for label, S in [("年末年始を除く", Rd.drop(index=[i for i in Rd.index if i in NY])),
                 ("2017年のみ", Rd[[str(i).startswith("2017") for i in Rd.index]])]:
    print(f"\n== 基準 best4 / {label} n={len(S)} ==")
    for col in Rd.columns:
        if col == "best4": continue
        d = (S[col] - S["best4"]).dropna(); se = d.std()/np.sqrt(len(d)); t = d.mean()/se
        ok = "★" if (100*(d<0).mean() >= 85 and d.median() <= d.mean()) or t <= -3 else ""
        print(f"  {col:18s} 差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}%  t {t:+.2f} {ok}")
