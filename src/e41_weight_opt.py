"""e41: 全脚の重み最適化。e39/e40 で保存した予測を使うので学習は不要。

脚: base / fam(family別33) / famgrp(6群) / type(5) / cluster(17) / sto(store別54)
★重みを27 fold で最適化して同じ27 fold で評価すると楽観的になるので、
  **leave-one-fold-out**（各 fold を除いた26 fold で重みを決め、その fold で評価）で汎化性能を測る。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import nnls
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, cv3

LEGS = ["base", "fam", "famgrp", "type", "cluster", "sto"]
data = {}
for fname, vs, ve in cv3.FOLDS:
    p39 = c.OUTPUT / f"e39_pred_{fname}.npz"; p40 = c.OUTPUT / f"e40_pred_{fname}.npz"
    if not (p39.exists() and p40.exists()): continue
    a, b = np.load(p39), np.load(p40)
    data[fname] = dict(y=a["y"], base=a["base"], fam=a["fam"], sto=a["sto"],
                       famgrp=b["famgrp"], type=b["type"], cluster=b["cluster"])
folds = list(data)
print(f"{len(folds)} fold の予測を読み込み")


def rmsle(p, y): return float(np.sqrt(np.mean((np.clip(p, 0, None) - y) ** 2)))


def fit_w(fs, legs):
    """指定 fold 群で非負最小二乗（重み和=1 に正規化）"""
    X = np.concatenate([np.column_stack([data[f][l] for l in legs]) for f in fs])
    y = np.concatenate([data[f]["y"] for f in fs])
    w, _ = nnls(X, y)
    return w / w.sum() if w.sum() > 0 else np.ones(len(legs)) / len(legs)


CANDS = {
    "現best (0.7base+0.3fam)": ("fixed", {"base": 0.7, "fam": 0.3}),
    "+famgrp.25": ("fixed", {"base": 0.525, "fam": 0.225, "famgrp": 0.25}),
    "NNLS base+fam+famgrp": ("loo", ["base", "fam", "famgrp"]),
    "NNLS +type": ("loo", ["base", "fam", "famgrp", "type"]),
    "NNLS 全6脚": ("loo", LEGS),
}
res = {}
for name, (mode, spec) in CANDS.items():
    per = {}
    for f in folds:
        if mode == "fixed":
            p = sum(v * data[f][k] for k, v in spec.items())
        else:
            others = [g for g in folds if g != f]          # leave-one-fold-out
            w = fit_w(others, spec)
            p = sum(wi * data[f][l] for wi, l in zip(w, spec))
        per[f] = rmsle(p, data[f]["y"])
    res[name] = per
R = pd.DataFrame(res)
R.to_csv(c.OUTPUT / "e41_weights.csv")

ref = "現best (0.7base+0.3fam)"
NY = ["2016-12-26"]
for label, S in [("全fold", R), ("年末年始を除く", R.drop(index=[i for i in R.index if i in NY])),
                 ("2017年のみ", R[[str(i).startswith("2017") for i in R.index]])]:
    print(f"\n== 基準 {ref} / {label} n={len(S)} ==")
    print(f"  {ref:24s} 平均 {S[ref].mean():.4f}")
    for col in R.columns:
        if col == ref: continue
        d = (S[col] - S[ref]).dropna(); se = d.std()/np.sqrt(len(d)); t = d.mean()/se
        ok = "★" if (100*(d<0).mean() >= 85 and d.median() <= d.mean()) or t <= -3 else ""
        print(f"  {col:24s} 平均 {S[col].mean():.4f}  差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}%  t {t:+.2f} {ok}")

print("\n== 全 fold で学習した参考重み（提出に使う値の目安）==")
for legs in [["base", "fam", "famgrp"], ["base", "fam", "famgrp", "type"], LEGS]:
    w = fit_w(folds, legs)
    print("  " + " / ".join(f"{l} {wi:.3f}" for l, wi in zip(legs, w)))
