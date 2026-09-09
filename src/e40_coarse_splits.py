"""e40: 「学習単位を変える」路線を粗い分割で追う。

e36/e39 で判明した構造:
  family別(1分割あたり5.1万行, 単体+0.015) → blend で 26/26 勝
  store別 (1分割あたり3.1万行, 単体+0.040) → 勝率0%、blend も無効
  = 分割が細かすぎて単体性能が落ちると、直交性があっても足せない（下限がある）

→ family別より**粗い/同等のデータ量**を保つ分割なら効くはず。
  cluster別(17): 約10万行 / type別(5): 約34万行 / family group別(6): 約28万行
  さらに family別と直交すれば **family blend の上に積める**ので、
  判定は base 比だけでなく **「現 best（base+family0.3）との差」** も測る。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv3
from config import PARAMS as _P
P = {**_P, "num_threads": 8, "seed": 42, "bagging_seed": 42, "feature_fraction_seed": 42}
ST, NR = pd.Timestamp("2015-01-01"), 500
df = F.build()

# family を売上規模とゼロ率でグループ化（6群）
_ref = df[(df.date >= "2016-01-01") & (df.date < "2017-01-01")]
_st = _ref.groupby("family", observed=True)["y"].agg(["mean", lambda s: (s <= 1e-9).mean()])
_st.columns = ["lvl", "zr"]
_st["grp"] = (pd.qcut(_st.lvl, 3, labels=[0, 1, 2]).astype(int) * 2
              + (_st.zr > 0.3).astype(int))
FAMGRP = _st["grp"].to_dict()
df["famgrp"] = df["family"].map(FAMGRP).astype("int8")
print("family group の内訳:", df.groupby("famgrp", observed=True)["family"].nunique().to_dict())

SPLITS = {
    "cluster": ("cluster", ["store_nbr", "city", "state", "type"]),   # cluster内で定数になる列を除く
    "type":    ("type", ["type"]),
    "famgrp":  ("famgrp", []),
}
rec = {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    m_va = (df.date >= vs) & (df.date <= ve) & df.sales.notna()
    if m_va.sum() == 0: continue
    m_tr = (df.date >= ST) & (df.date < vs) & df.sales.notna()
    tr, va = df[m_tr], df[m_va]
    yv = va["y"].values
    d = np.load(c.OUTPUT / f"e39_pred_{fname}.npz")     # e39 の base/fam 予測を再利用
    base, fam = d["base"], d["fam"]
    cur = 0.7 * base + 0.3 * fam                        # 現 best（LB 0.41889 と同じ構成）
    row = {"base": float(np.sqrt(np.mean((base - yv) ** 2))),
           "cur(b+fam.3)": float(np.sqrt(np.mean((cur - yv) ** 2)))}

    preds = {}
    for tag, (key, drop_cols) in SPLITS.items():
        feats = [f for f in F.FEATURES if f not in drop_cols]
        cats = [x for x in F.CATS if x in feats]
        p = np.zeros(len(va)); vk = va[key].values
        for k_, g in tr.groupby(key, observed=True):
            m = vk == k_
            if m.sum() == 0 or len(g) < 20000: continue
            p[m] = np.clip(lgb.train(P, lgb.Dataset(g[feats], g["y"], categorical_feature=cats),
                                     num_boost_round=NR).predict(va[m][feats]), 0, None)
        miss = p == 0
        if miss.any(): p[miss] = base[miss]
        preds[tag] = p
        row[f"{tag}_solo"] = float(np.sqrt(np.mean((p - yv) ** 2)))
        row[f"b+{tag}.3"] = float(np.sqrt(np.mean((0.7 * base + 0.3 * p - yv) ** 2)))
        # 現 best の上に積む（family blend と直交するか）
        for w in [0.15, 0.25]:
            row[f"cur+{tag}{w}"] = float(np.sqrt(np.mean(((1 - w) * cur + w * p - yv) ** 2)))
    np.savez_compressed(c.OUTPUT / f"e40_pred_{fname}.npz", **preds)
    rec[fname] = row
    print(f"{fname}  " + "  ".join(f"{k} {v:.4f}" for k, v in row.items() if k.endswith("solo") or k.startswith("cur"))
          + f"  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec).T
R.to_csv(c.OUTPUT / "e40_coarse.csv")
print(f"\n== {len(R)} fold 平均 ==")
print(R.mean().round(4).to_string())
NY = [i for i in R.index if i == "2016-12-26"]
for label, S in [("全fold", R), ("年末年始を除く", R.drop(index=NY)), ("2017年のみ", R[[str(i).startswith("2017") for i in R.index]])]:
    for ref in ["base", "cur(b+fam.3)"]:
        print(f"\n== 基準 {ref} / {label} n={len(S)} ==")
        for col in R.columns:
            if col in ("base", "cur(b+fam.3)"): continue
            if ref == "cur(b+fam.3)" and not col.startswith("cur+"): continue
            if ref == "base" and col.startswith("cur+"): continue
            dd = (S[col] - S[ref]).dropna(); se = dd.std()/np.sqrt(len(dd)); t = dd.mean()/se
            ok = "★" if (100*(dd<0).mean() >= 85 and dd.median() <= dd.mean()) or t <= -3 else ""
            print(f"  {col:16s} 平均差 {dd.mean():+.4f}  中央値 {dd.median():+.4f}  勝率 {100*(dd<0).mean():3.0f}%  t {t:+.2f} {ok}")
