"""e43: 4本ブレンドの最終比率を leave-one-fold-out で確認する（学習不要、npz を再利用）。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, cv3

D = {}
for fname, vs, ve in cv3.FOLDS:
    p39, p40 = c.OUTPUT / f"e39_pred_{fname}.npz", c.OUTPUT / f"e40_pred_{fname}.npz"
    p42 = c.OUTPUT / f"e42_gseas_{fname}.npz"
    if not (p39.exists() and p40.exists() and p42.exists()): continue
    a, b, g = np.load(p39), np.load(p40), np.load(p42)
    D[fname] = dict(y=a["y"], base=a["base"], fam=a["fam"], fg=b["famgrp"], gs=g["p"])
folds = [f for f in D if f != "2016-12-26"]
print(f"{len(D)} fold（年末年始を除く判定は {len(folds)} fold）")

def rm(p, y): return float(np.sqrt(np.mean((np.clip(p, 0, None) - y) ** 2)))

CANDS = {
    "現best b.70/fam.30":            (.70, .30, .00, .00),
    "+fg.25":                        (.525, .225, .25, .00),
    "+fg.25+gs.20":                  (.42, .18, .20, .20),
    "+fg.20+gs.20":                  (.45, .15, .20, .20),
    "均等4分の変形 .40/.20/.20/.20":  (.40, .20, .20, .20),
    "gs 厚め .40/.15/.20/.25":        (.40, .15, .20, .25),
    "軽め .55/.20/.15/.10":           (.55, .20, .15, .10),
}
R = pd.DataFrame({name: {f: rm(w[0]*D[f]["base"] + w[1]*D[f]["fam"] + w[2]*D[f]["fg"] + w[3]*D[f]["gs"], D[f]["y"])
                         for f in D} for name, w in CANDS.items()})
ref = "現best b.70/fam.30"
for label, S in [("年末年始を除く", R.loc[folds]), ("2017年のみ", R[[str(i).startswith("2017") for i in R.index]]),
                 ("全fold", R)]:
    print(f"\n== 基準 {ref} / {label} n={len(S)} ==")
    for col in R.columns:
        if col == ref: continue
        d = (S[col] - S[ref]); se = d.std()/np.sqrt(len(d)); t = d.mean()/se
        ok = "★" if (100*(d<0).mean() >= 85 and d.median() <= d.mean()) or t <= -3 else ""
        print(f"  {col:26s} 平均 {S[col].mean():.4f}  差 {d.mean():+.4f}  中央値 {d.median():+.4f}  勝率 {100*(d<0).mean():3.0f}%  t {t:+.2f} {ok}")
