"""e08: early stopping の楽観バイアスを測る（i20）。

e02 は fold 内 valid で early stopping しているため、best_iter の選択分だけ CV が楽観化する。
固定 iter で回した CV との差が、その楽観幅そのもの。以降すべての判定の較正値になる。
usage: python3 e08_fixed_iter.py [iters comma-separated]
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F
from config import PARAMS as _P, TRAIN_START as _TS
PARAMS = {**_P, "num_threads": 6}
ITERS = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "900").split(",")]
TRAIN_START = pd.Timestamp(_TS)

df = F.build()
folds = c.make_folds(5)
# fold ごとに最大 iter まで1回学習し、各 iter 時点の予測を取り出す（学習は1回で済む）
res = {n: [] for n in ITERS}
for k, (tr_end, vs, ve) in enumerate(folds):
    t0 = time.time()
    m_tr = (df.date >= TRAIN_START) & (df.date < tr_end) & df.sales.notna()
    m_va = (df.date >= vs) & (df.date <= ve)
    mdl = lgb.train(PARAMS, lgb.Dataset(df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"],
                                        categorical_feature=F.CATS), num_boost_round=max(ITERS))
    yva = df.loc[m_va, "y"].values
    line = []
    for n in ITERS:
        p = np.clip(mdl.predict(df.loc[m_va, F.FEATURES], num_iteration=n), 0, None)
        s = float(np.sqrt(np.mean((p - yva) ** 2)))
        res[n].append(s); line.append(f"{n}:{s:.4f}")
    print(f"fold{k} " + "  ".join(line) + f"  {time.time()-t0:.0f}s", flush=True)

print("\n== 固定 iter 別 CV（early stopping 版 e02 = 0.3856）==")
for n in ITERS:
    print(f"  iter {n:5d}  {np.mean(res[n]):.4f}  (e02比 {np.mean(res[n])-0.3856:+.4f})  folds {[round(x,4) for x in res[n]]}")
best = min(ITERS, key=lambda n: np.mean(res[n]))
print(f"\n最良固定 iter = {best} ({np.mean(res[best]):.4f})")
print(f"early stopping の楽観幅 = {0.3856 - np.mean(res[best]):+.4f}  （負なら early stopping が楽観化していた）")
