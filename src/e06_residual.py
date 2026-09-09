"""e06: 残差学習。目的を y から (y - rmean7) に変える。他は e02 と同一。

動機: e02 は gain の 84% を rmean7/rmean14 による「系列の水準合わせ」に費やしている。
水準をオフセットとして外に出せば、木の容量を曜日・祝日・プロモ・季節性の構造に回せるはず。
usage: python3 e06_residual.py [keep_rmean7: 1/0] [TRAIN_START]
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F
from config import PARAMS as _P, TRAIN_START as _TS
PARAMS = {**_P, 'num_threads': 6}

KEEP = bool(int(sys.argv[1])) if len(sys.argv) > 1 else True
TRAIN_START = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else _TS)

df = F.build()
df["y_res"] = (df["y"] - df["rmean7"]).astype("float32")
feats = F.FEATURES if KEEP else [f for f in F.FEATURES if f != "rmean7"]

sc, it = [], []
for k, (tr_end, vs, ve) in enumerate(c.make_folds(5)):
    t0 = time.time()
    m_tr = (df.date >= TRAIN_START) & (df.date < tr_end) & df.sales.notna() & df.y_res.notna()
    m_va = (df.date >= vs) & (df.date <= ve)
    ds = lgb.Dataset(df.loc[m_tr, feats], df.loc[m_tr, "y_res"], categorical_feature=F.CATS)
    # early stopping も残差空間の RMSE で行う（オフセットは valid でも既知なので順位は変わらない）
    mdl = lgb.train(PARAMS, ds, num_boost_round=3000,
                    valid_sets=[lgb.Dataset(df.loc[m_va, feats], df.loc[m_va, "y_res"],
                                            categorical_feature=F.CATS, reference=ds)],
                    callbacks=[lgb.early_stopping(100, verbose=False)])
    p = mdl.predict(df.loc[m_va, feats], num_iteration=mdl.best_iteration) + df.loc[m_va, "rmean7"].values
    p = np.clip(p, 0, None)
    s = float(np.sqrt(np.mean((p - df.loc[m_va, "y"].values) ** 2)))
    sc.append(s); it.append(mdl.best_iteration)
    print(f"fold{k} RMSLE {s:.4f}  best_iter {mdl.best_iteration}  {time.time()-t0:.0f}s", flush=True)

print(f"\n== e06 残差学習 (keep_rmean7={KEEP}, start={TRAIN_START.date()}) mean {np.mean(sc):.4f} ==")
print(f"   folds {[round(x,4) for x in sc]}  iters {it}   e02=0.3856 比 {np.mean(sc)-0.3856:+.4f}")
imp = pd.Series(mdl.feature_importance("gain"), index=feats).sort_values(ascending=False)
print("\n== 重要度 top15 ==")
print((imp / imp.sum()).head(15).round(4).to_string())
