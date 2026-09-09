"""e07: expanding TE のキーを粗いものだけに絞る。e05(全11キー, 0.3865)の追試。

仮説: (store,family,*) の細かいキーは rmean7/rmean14/dow4_mean と情報が重複し、
ノイズだけを持ち込んでいる。family/cluster レベルの粗いキーなら重複が少ないはず。
元の e05 の docstring: expanding(過去のみ)TE。e04(素朴TE=自己リーク版, 0.3906)の修正。他は e02 と同一（同一 params / 同一 fold / 同一ベース特徴）。
usage: python3 e04_te.py [TRAIN_START] [level_days] [m]
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, te as TE
from config import PARAMS as _P
PARAMS = {**_P, 'num_threads': 5}
COARSE = [k for k in TE.TE_KEYS if 'store_nbr' not in k[1]]  # store x family の細かいキーを除外  # e03 と並行実行するためスレッドを分ける

TRAIN_START = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2015-01-01")
LEVEL_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 365
M = int(sys.argv[3]) if len(sys.argv) > 3 else 20

df = F.build()
folds = c.make_folds(5)
sc, it, imps = [], [], []
for k, (tr_end, vs, ve) in enumerate(folds):
    t0 = time.time()
    # TE は fold の train 期間のみで作る（valid の情報は一切使わない）
    te_mask = (df.date >= TRAIN_START) & (df.date < tr_end) & df.sales.notna()
    te_df, te_cols = TE.add_te(df, te_mask, level_days=LEVEL_DAYS, m=M, keys=COARSE)
    X = pd.concat([df[F.FEATURES], te_df], axis=1)
    feats = F.FEATURES + te_cols

    m_tr = te_mask
    m_va = (df.date >= vs) & (df.date <= ve)
    ds = lgb.Dataset(X[m_tr], df.loc[m_tr, "y"], categorical_feature=F.CATS)
    mdl = lgb.train(PARAMS, ds, num_boost_round=3000,
                    valid_sets=[lgb.Dataset(X[m_va], df.loc[m_va, "y"], categorical_feature=F.CATS, reference=ds)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    p = np.clip(mdl.predict(X[m_va], num_iteration=mdl.best_iteration), 0, None)
    s = float(np.sqrt(np.mean((p - df.loc[m_va, "y"].values) ** 2)))
    sc.append(s); it.append(mdl.best_iteration)
    imps.append(pd.Series(mdl.feature_importance("gain"), index=feats))
    print(f"fold{k} RMSLE {s:.4f}  best_iter {mdl.best_iteration}  {time.time()-t0:.0f}s", flush=True)

print(f"\n== e07 TE(coarse keys) (start={TRAIN_START.date()}, level_days={LEVEL_DAYS}, m={M}) mean {np.mean(sc):.4f} ==")
print(f"   folds {[round(x,4) for x in sc]}  iters {it}")
imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
print("\n== 重要度 top20 (gain 比) ==")
print((imp / imp.sum()).head(20).round(4).to_string())
print("\n== TE 列の合計 gain 比: %.4f ==" % (imp[[x for x in imp.index if x.startswith('te_')]].sum() / imp.sum()))
