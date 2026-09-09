"""e01b: dow4_mean の誤差分解。fold0(7/31-8/15) が他foldより悪い原因を特定する。"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

tr = pd.read_csv(c.DATA/"train.csv", parse_dates=["date"])
piv = tr.pivot_table(index="date", columns=["store_nbr","family"], values="sales", aggfunc="mean")
piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max()))
L = np.log1p(piv)

def dow4_mean(hist, vdates):
    tail = hist.tail(28); tbl = tail.groupby(tail.index.dayofweek).mean()
    return np.stack([tbl.loc[d.dayofweek].fillna(0.0).values for d in vdates])

folds = c.make_folds(5)
recs = []
for k,(te_,vs,ve) in enumerate(folds):
    hist = L[L.index < te_]; vd = pd.date_range(vs,ve)
    p = np.clip(dow4_mean(hist,vd),0,None); y = L.reindex(vd).values
    se = (p-y)**2
    for j,(s,f) in enumerate(piv.columns):
        recs.append(dict(fold=k, store=s, family=f, sse=se[:,j].sum(), n=len(vd)))
    if k==0:
        d0 = pd.DataFrame(se, index=vd, columns=piv.columns)
df = pd.DataFrame(recs)

print("== family別 RMSLE: fold0 vs fold1-4平均（悪化幅の大きい順 上位12）==")
t = df.groupby(["family","fold"]).sse.sum().unstack()
n_cells = 54*16
r = np.sqrt(t/n_cells)
cmp = pd.DataFrame({"fold0": r[0], "fold1_4": r[[1,2,3,4]].mean(axis=1)})
cmp["diff"] = cmp.fold0 - cmp.fold1_4
cmp["fold0_SSE比"] = (t[0]/t[0].sum()).round(3)
print(cmp.sort_values("diff", ascending=False).head(12).round(4).to_string())

print("\n== fold0 の日別 RMSLE ==")
print(np.sqrt(d0.sum(axis=1)/1782).round(4).to_string())

print("\n== SCHOOL AND OFFICE SUPPLIES の月別 総売上（年×月）==")
sc = tr[tr.family=="SCHOOL AND OFFICE SUPPLIES"].groupby([tr.date.dt.year, tr.date.dt.month]).sales.sum().unstack()
print(sc.round(0).to_string())

print("\n== fold0 SSE 寄与 上位10 family ==")
print((t[0].sort_values(ascending=False).head(10)/t[0].sum()).round(3).to_string())
