"""e03: 学習期間の長さを振る。他は e02 と同一（同一 params / 同一特徴 / 同一 fold）。"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F
from config import PARAMS

STARTS = ["2013-01-01", "2014-04-01", "2015-01-01", "2016-01-01", "2016-08-16", "2017-01-01"]
df = F.build()
folds = c.make_folds(5)
res = {}
for st in STARTS:
    st_ts = pd.Timestamp(st)
    sc, it = [], []
    t0 = time.time()
    for k, (tr_end, vs, ve) in enumerate(folds):
        m_tr = (df.date >= st_ts) & (df.date < tr_end) & df.sales.notna()
        m_va = (df.date >= vs) & (df.date <= ve)
        Xtr, ytr = df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"]
        Xva, yva = df.loc[m_va, F.FEATURES], df.loc[m_va, "y"]
        ds = lgb.Dataset(Xtr, ytr, categorical_feature=F.CATS)
        mdl = lgb.train(PARAMS, ds, num_boost_round=3000,
                        valid_sets=[lgb.Dataset(Xva, yva, categorical_feature=F.CATS, reference=ds)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        p = np.clip(mdl.predict(Xva, num_iteration=mdl.best_iteration), 0, None)
        sc.append(float(np.sqrt(np.mean((p - yva.values) ** 2)))); it.append(mdl.best_iteration)
    res[st] = (np.mean(sc), sc, it, int(m_tr.sum()))
    print(f"{st}  mean {np.mean(sc):.4f}  folds {[round(x,4) for x in sc]}  iters {it}  n_tr(last fold) {m_tr.sum():,}  {time.time()-t0:.0f}s", flush=True)

print("\n== 学習開始日 別 CV RMSLE ==")
for st, (m, sc, it, n) in sorted(res.items(), key=lambda x: x[1][0]):
    print(f"  {st}  {m:.4f}  (e02=2015-01-01 比 {m-res['2015-01-01'][0]:+.4f})")
