"""e12(=e09の続き): 直近重み付け（i22）。

e03 で「期間カットは害」と出たが、それは古いデータを捨てたから。データは全部使いつつ
直近を重くすれば、分布整合（特に test の onpromotion 平均が train の2.7倍という差）と
データ量を両立できるはず。weight = 0.5 ** (age_days / half_life)。
usage: python3 e09_recency_weight.py [half_lives comma-separated, 0=重みなし]
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F
from config import PARAMS as _P
PARAMS = {**_P, "num_threads": 5}
HALF = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "180,365,730").split(",")]
TRAIN_START = pd.Timestamp("2014-04-01")  # e03 の最良

df = F.build()
folds = c.make_folds(5)
for hl in HALF:
    sc, it = [], []
    for k, (tr_end, vs, ve) in enumerate(folds):
        m_tr = (df.date >= TRAIN_START) & (df.date < tr_end) & df.sales.notna()
        m_va = (df.date >= vs) & (df.date <= ve)
        if hl > 0:
            age = (tr_end - df.loc[m_tr, "date"]).dt.days.values
            w = 0.5 ** (age / hl)
        else:
            w = None
        ds = lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"], weight=w, categorical_feature=F.CATS)
        mdl = lgb.train(PARAMS, ds, num_boost_round=3000,
                        valid_sets=[lgb.Dataset(df.loc[m_va, F.FEATURES], df.loc[m_va, "y"],
                                                categorical_feature=F.CATS, reference=ds)],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        p = np.clip(mdl.predict(df.loc[m_va, F.FEATURES], num_iteration=mdl.best_iteration), 0, None)
        sc.append(float(np.sqrt(np.mean((p - df.loc[m_va, "y"].values) ** 2)))); it.append(mdl.best_iteration)
    tag = f"half_life={hl}d" if hl else "重みなし(2014-04起点)"
    print(f"{tag:26s} mean {np.mean(sc):.4f}  folds {[round(x,4) for x in sc]}  iters {it}", flush=True)
