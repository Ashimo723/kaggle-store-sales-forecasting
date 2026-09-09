"""e25: two-part (hurdle) model。E[log1p(y)] = P(y>0) x E[log1p(y)|y>0] に分解する。

e24: 「y==0 を完璧に当てる」oracle は -0.0377（全施策合計の11倍）。
現行は1本の回帰で「売れるか」と「いくら売れるか」を同時推定しており、この分解は未検証。

測るもの:
 1. 分類器 P(y>0) の AUC（そもそも 0/非0 が予測可能か）
 2. two-part 予測 p*m の RMSLE
 3. 現行予測とのブレンド
 4. p が低い行だけ現行予測を縮小する後処理（分解より軽い介入）
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv2
from config import PARAMS as _P
PR = {**_P, "num_threads": 8}
PC = {**_P, "objective": "binary", "metric": "auc", "num_threads": 8}
TRAIN_START, HL = pd.Timestamp("2014-04-01"), 365
df = F.build()

res = {}
for fl, vs, ve, drop in cv2.FOLDS:
    t0 = time.time()
    m_tr = (df.date >= TRAIN_START) & (df.date < vs) & df.sales.notna()
    m_va = (df.date >= vs) & (df.date <= ve) & (~df.store_nbr.isin(drop) if drop else True)
    w = 0.5 ** ((vs - df.loc[m_tr, "date"]).dt.days.values / HL)
    ytr, yva = df.loc[m_tr, "y"], df.loc[m_va, "y"].values

    # (1) 現行: 直接回帰
    base = lgb.train(PR, lgb.Dataset(df.loc[m_tr, F.FEATURES], ytr, weight=w,
                                     categorical_feature=F.CATS), num_boost_round=900)
    p_base = np.clip(base.predict(df.loc[m_va, F.FEATURES]), 0, None)

    # (2) 分類器 P(y>0)
    clf = lgb.train(PC, lgb.Dataset(df.loc[m_tr, F.FEATURES], (ytr > 1e-9).astype(int), weight=w,
                                    categorical_feature=F.CATS), num_boost_round=900)
    p_pos = clf.predict(df.loc[m_va, F.FEATURES])
    auc = roc_auc_score((yva > 1e-9).astype(int), p_pos)

    # (3) 条件付き回帰 E[log1p(y) | y>0]
    m_tr_pos = m_tr & (df.y > 1e-9)
    w_pos = 0.5 ** ((vs - df.loc[m_tr_pos, "date"]).dt.days.values / HL)
    reg = lgb.train(PR, lgb.Dataset(df.loc[m_tr_pos, F.FEATURES], df.loc[m_tr_pos, "y"], weight=w_pos,
                                    categorical_feature=F.CATS), num_boost_round=900)
    m_cond = np.clip(reg.predict(df.loc[m_va, F.FEATURES]), 0, None)

    r = {"現行(直接回帰)": p_base, "two-part p*m": p_pos * m_cond}
    for a in [0.3, 0.5, 0.7]:
        r[f"blend {a:.1f}"] = a * (p_pos * m_cond) + (1 - a) * p_base
    # 後処理: p が低い行だけ現行予測を縮小（分解より軽い介入）
    for th in [0.3, 0.5]:
        for sh in [0.5, 0.0]:
            q = p_base.copy(); q[p_pos < th] *= sh
            r[f"p<{th} を x{sh}"] = q
    out = {k: float(np.sqrt(np.mean((np.clip(v, 0, None) - yva) ** 2))) for k, v in r.items()}
    out["_auc"] = auc
    res[fl] = out
    print(f"{fl}  AUC {auc:.4f}  " + "  ".join(f"{k} {v:.4f}" for k, v in out.items() if not k.startswith("_")) + f"  {time.time()-t0:.0f}s", flush=True)

print("\n== 2fold 平均（現行比）==")
keys = [k for k in res[list(res)[0]] if not k.startswith("_")]
b = np.mean([res[f]["現行(直接回帰)"] for f in res])
for k in keys:
    vals = [res[f][k] for f in res]
    m = np.mean(vals)
    da = res["A:2017-08前半"][k] - res["A:2017-08前半"]["現行(直接回帰)"]
    db = res["B:2016-08後半"][k] - res["B:2016-08後半"]["現行(直接回帰)"]
    tag = "★改善" if (da < 0 and db < 0 and abs(m - b) >= 0.002) else ("同方向だが小" if (da < 0) == (db < 0) else "符号不一致")
    print(f"  {k:18s} {m:.4f}  ({m-b:+.4f})  A {da:+.4f} B {db:+.4f}  {tag}")
print(f"\n分類器 AUC: " + " / ".join(f"{f} {res[f]['_auc']:.4f}" for f in res))
print("参考: e24 の oracle（y==0 を完璧に 0 にする）= -0.0377")
