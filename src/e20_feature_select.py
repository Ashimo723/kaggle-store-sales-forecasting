"""e20: 特徴選択。重要度の下位カット / 上位カット を両方測る。

e18 で「特徴追加は全滅」と分かったので、逆に削る方向を試す。
 - 下位カット: ノイズ列を落として汎化を上げる（標準的な特徴選択）
 - 上位カット: gain の 84% を占める rmean7/rmean14 を外し、木が他の構造を学ぶか見る
   （e06 の残差学習は「オフセットとして外に出す」形で失敗したが、単純に削るのは未検証）
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv2
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
TRAIN_START, HL = pd.Timestamp("2014-04-01"), 365
df = F.build()


def run(feats, seeds=(42, 1)):
    """2fold x 複数 seed（e16: seed 間 sd 0.0011 なので単一 seed では判定できない）"""
    per = {}
    for fl, vs, ve, drop in cv2.FOLDS:
        m_tr = (df.date >= TRAIN_START) & (df.date < vs) & df.sales.notna()
        m_va = (df.date >= vs) & (df.date <= ve) & (~df.store_nbr.isin(drop) if drop else True)
        w = 0.5 ** ((vs - df.loc[m_tr, "date"]).dt.days.values / HL)
        yv = df.loc[m_va, "y"].values
        ps = []
        for sd in seeds:
            mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                            lgb.Dataset(df.loc[m_tr, feats], df.loc[m_tr, "y"], weight=w,
                                        categorical_feature=[x for x in F.CATS if x in feats]),
                            num_boost_round=900)
            ps.append(np.clip(mdl.predict(df.loc[m_va, feats]), 0, None))
        per[fl] = float(np.sqrt(np.mean((np.mean(ps, axis=0) - yv) ** 2)))
        if fl.startswith("A"):
            per["_imp"] = pd.Series(mdl.feature_importance("gain"), index=feats)
    return per


base = run(F.FEATURES)
imp = (base.pop("_imp")).sort_values(ascending=False)
bmean = np.mean(list(base.values()))
print(f"ベース56特徴  A {base['A:2017-08前半']:.4f}  B {base['B:2016-08後半']:.4f}  平均 {bmean:.4f}\n")
print("== 重要度（gain比）上位10 / 下位10 ==")
print((imp / imp.sum()).head(10).round(4).to_string())
print("  ...")
print((imp / imp.sum()).tail(10).round(5).to_string())

rows = []
print("\n== 下位カット ==")
for n in [10, 20, 30, 40]:
    feats = [f for f in F.FEATURES if f in imp.index[:len(imp) - n]]
    r = run(feats); r.pop("_imp", None)
    m = np.mean(list(r.values()))
    da, db = r["A:2017-08前半"] - base["A:2017-08前半"], r["B:2016-08後半"] - base["B:2016-08後半"]
    tag = "★改善" if (da < 0 and db < 0 and abs(m - bmean) >= 0.002) else ("同方向だが小" if (da < 0) == (db < 0) else "符号不一致")
    rows.append((f"下位{n}カット({len(feats)}特徴)", r, m, da, db, tag))
    print(f"下位{n}カット ({len(feats):2d}特徴)  A {r['A:2017-08前半']:.4f}({da:+.4f})  B {r['B:2016-08後半']:.4f}({db:+.4f})  平均 {m:.4f}({m-bmean:+.4f})  {tag}", flush=True)

print("\n== 上位カット ==")
for k in [1, 2, 3, 5]:
    feats = [f for f in F.FEATURES if f not in set(imp.index[:k])]
    r = run(feats); r.pop("_imp", None)
    m = np.mean(list(r.values()))
    da, db = r["A:2017-08前半"] - base["A:2017-08前半"], r["B:2016-08後半"] - base["B:2016-08後半"]
    tag = "★改善" if (da < 0 and db < 0 and abs(m - bmean) >= 0.002) else ("同方向だが小" if (da < 0) == (db < 0) else "符号不一致")
    print(f"上位{k}カット [{', '.join(imp.index[:k])}]  A {r['A:2017-08前半']:.4f}({da:+.4f})  B {r['B:2016-08後半']:.4f}({db:+.4f})  平均 {m:.4f}({m-bmean:+.4f})  {tag}", flush=True)
