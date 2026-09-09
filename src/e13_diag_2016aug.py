"""e13: 2016-08後半が 0.6763（同年8月前半 0.4335）と極端に悪い原因を特定する。

これが「毎年8月後半に起きる構造」なら CV に入れるべきだが、
「2016年固有の異常」なら入れると判定が歪む。採用可否はこの診断で決める。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F
from config import PARAMS as _P
PARAMS = {**_P, "num_threads": 6}

df = F.build()
out = {}
for label, vs, ve in [("2016後半", "2016-08-16", "2016-08-31"), ("2016前半", "2016-07-31", "2016-08-15")]:
    vs, ve = pd.Timestamp(vs), pd.Timestamp(ve)
    m_tr = (df.date >= "2013-01-01") & (df.date < vs) & df.sales.notna()
    m_va = (df.date >= vs) & (df.date <= ve)
    mdl = lgb.train(PARAMS, lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"],
                                        categorical_feature=F.CATS), num_boost_round=900)
    v = df.loc[m_va, ["date", "store_nbr", "family", "y", "sales"]].copy()
    v["pred"] = np.clip(mdl.predict(df.loc[m_va, F.FEATURES]), 0, None)
    v["se"] = (v.pred - v.y) ** 2
    v["signed"] = v.pred - v.y
    out[label] = v
    print(f"{label} RMSLE {np.sqrt(v.se.mean()):.4f}", flush=True)

v = out["2016後半"]
print("\n== 日別 RMSLE（2016-08後半）==")
print(v.groupby("date").se.mean().pow(0.5).round(4).to_string())
print("\n== 店舗別 SSE 寄与 上位10 ==")
t = v.groupby("store_nbr").agg(sse=("se", "sum"), rmsle=("se", lambda x: np.sqrt(x.mean())),
                               bias=("signed", "mean"))
t["SSE比"] = t.sse / t.sse.sum()
print(t.sort_values("SSE比", ascending=False).head(10)[["rmsle", "bias", "SSE比"]].round(4).to_string())
print("\n== family別 SSE 寄与 上位8 ==")
tf = v.groupby("family", observed=True).agg(rmsle=("se", lambda x: np.sqrt(x.mean())), sse=("se", "sum"))
tf["SSE比"] = tf.sse / tf.sse.sum()
print(tf.sort_values("SSE比", ascending=False).head(8)[["rmsle", "SSE比"]].round(4).to_string())
print("\n== 予測 > 実績（過大予測）の割合と、実績0の行の割合 ==")
print(f"  過大予測 {100*(v.signed>0).mean():.1f}%   実績 sales==0 の行 {100*(v.sales==0).mean():.1f}%")
print(f"  実績0 の行の SSE 寄与: {100*v.loc[v.sales==0,'se'].sum()/v.se.sum():.1f}%")
print("\n== 上位寄与店舗の日別 実績合計（0 なら休業）==")
top = t.sort_values("SSE比", ascending=False).head(3).index.tolist()
piv = v[v.store_nbr.isin(top)].pivot_table(index="date", columns="store_nbr", values="sales", aggfunc="sum")
print(piv.round(0).to_string())
print("\n== 参考: 同店舗の 2016-08前半 日別実績合計 ==")
vp = out["2016前半"]
print(vp[vp.store_nbr.isin(top)].pivot_table(index="date", columns="store_nbr", values="sales", aggfunc="sum").round(0).to_string())
