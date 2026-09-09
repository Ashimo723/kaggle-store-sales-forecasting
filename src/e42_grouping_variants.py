"""e42: family の集約軸を変える。

e40 の発見: 店舗側の切り口(store/cluster/type)より **商品側(family/famgrp)が有効**。
  famgrp(売上規模3 × ゼロ率2 = 6群, 28万行/群) は単体でも base を上回った(-0.0016)。
→ 「何で集約するか」で更に良い切り口があるはず。群数と集約基準を振る。

集約軸:
  g6   : 売上規模3 × ゼロ率2（現行、採用済み）
  g4   : 売上規模2 × ゼロ率2
  g12  : 売上規模3 × ゼロ率2 × 年次季節性2
  gseas: 年次季節性3 × ゼロ率2（季節性の形で切る）
  gpromo: プロモ反応3 × ゼロ率2（プロモ弾力性で切る）
判定は base 比だけでなく **cur2 = 現best+famgrp0.25 の上に積めるか**も測る。
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

# --- family の統計量（2016年で算出。valid の情報は使わない）---
ref = df[(df.date >= "2016-01-01") & (df.date < "2017-01-01")]
st = ref.groupby("family", observed=True).agg(
    lvl=("y", "mean"), zr=("y", lambda s: (s <= 1e-9).mean()))
# 年次季節性の強さ: 月別平均の変動係数
mo = ref.groupby(["family", ref.date.dt.month], observed=True)["y"].mean().unstack()
st["seas"] = (mo.std(axis=1) / (mo.mean(axis=1).abs() + 1e-6))
# プロモ弾力性: onpromotion>0 の日と =0 の日の平均差
pr = ref.assign(pf=(ref.onpromotion > 0).astype(int)).groupby(["family", "pf"], observed=True)["y"].mean().unstack()
st["promo"] = (pr[1] - pr[0]).fillna(0) if 1 in pr.columns else 0.0

def q(s, n): return pd.qcut(s.rank(method="first"), n, labels=range(n)).astype(int)
GRPS = {
    "g6":     q(st.lvl, 3) * 2 + (st.zr > .3).astype(int),
    "g4":     q(st.lvl, 2) * 2 + (st.zr > .3).astype(int),
    "g12":    (q(st.lvl, 3) * 2 + (st.zr > .3).astype(int)) * 2 + q(st.seas, 2),
    "gseas":  q(st.seas, 3) * 2 + (st.zr > .3).astype(int),
    "gpromo": q(st.promo, 3) * 2 + (st.zr > .3).astype(int),
}
for k, v in GRPS.items():
    df[k] = df["family"].map(v.to_dict()).astype("int16")
    print(f"{k}: {v.nunique()} 群  内訳 {v.value_counts().sort_index().to_dict()}")

rec = {}
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    p39 = c.OUTPUT / f"e39_pred_{fname}.npz"; p40 = c.OUTPUT / f"e40_pred_{fname}.npz"
    if not (p39.exists() and p40.exists()): continue
    a, b = np.load(p39), np.load(p40)
    yv, base, fam, fg = a["y"], a["base"], a["fam"], b["famgrp"]
    cur = 0.7 * base + 0.3 * fam
    cur2 = 0.75 * cur + 0.25 * fg                     # 現時点の最良（+famgrp.25）
    m_va = (df.date >= vs) & (df.date <= ve) & df.sales.notna()
    m_tr = (df.date >= ST) & (df.date < vs) & df.sales.notna()
    tr, va = df[m_tr], df[m_va]
    r = lambda p: float(np.sqrt(np.mean((np.clip(p, 0, None) - yv) ** 2)))
    row = {"base": r(base), "cur": r(cur), "cur2": r(cur2)}
    for tag in GRPS:
        p = np.zeros(len(va)); vk = va[tag].values
        for k_, g in tr.groupby(tag, observed=True):
            m = vk == k_
            if m.sum() == 0 or len(g) < 20000: continue
            p[m] = np.clip(lgb.train(P, lgb.Dataset(g[F.FEATURES], g["y"], categorical_feature=F.CATS),
                                     num_boost_round=NR).predict(va[m][F.FEATURES]), 0, None)
        miss = p == 0
        if miss.any(): p[miss] = base[miss]
        row[f"{tag}_solo"] = r(p)
        row[f"cur+{tag}.25"] = r(0.75 * cur + 0.25 * p)
        row[f"cur2+{tag}.20"] = r(0.80 * cur2 + 0.20 * p)
        np.savez_compressed(c.OUTPUT / f"e42_{tag}_{fname}.npz", p=p)
    rec[fname] = row
    print(f"{fname}  " + "  ".join(f"{k.replace('_solo','')} {row[k]:.4f}" for k in row if k.endswith('_solo')) + f"  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rec).T
R.to_csv(c.OUTPUT / "e42_grouping.csv")
NY = ["2016-12-26"]
for label, S in [("年末年始を除く", R.drop(index=[i for i in R.index if i in NY])),
                 ("2017年のみ", R[[str(i).startswith("2017") for i in R.index]])]:
    for ref_col in ["cur", "cur2"]:
        print(f"\n== 基準 {ref_col} / {label} n={len(S)} ==")
        for col in R.columns:
            if col in ("base", "cur", "cur2"): continue
            if ref_col == "cur" and not col.startswith("cur+"): continue
            if ref_col == "cur2" and not col.startswith("cur2+"): continue
            d = (S[col] - S[ref_col]).dropna(); se = d.std()/np.sqrt(len(d)); t = d.mean()/se
            ok = "★" if (100*(d<0).mean() >= 85 and d.median() <= d.mean()) or t <= -3 else ""
            print(f"  {col:18s} 差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}%  t {t:+.2f} {ok}")
