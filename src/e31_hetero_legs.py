"""e31: 異質な脚のアンサンブル（cv3 27fold、|t|>=3 基準）。

動機: ここまで潰した軸（特徴/ハイパラ/目的変数/後処理）は全て単一 LightGBM の内部変更で、
e20 で「特徴が冗長で情報が飽和」と確認済み。飽和を破るには**情報源の違う予測器**が要る。

脚:
  L = LightGBM 3seed（現 best、LB 0.42468）
  R = 系列別 Ridge（ラグを一切使わない: 曜日/年次フーリエ/給与日/祝日/プロモ/トレンド）
  D = 同曜日4週平均（e01 の best ベースライン）
  N = 前年同期（lag364）
  C = Croston 法（間欠需要向け。e24 のゼロ率「中」帯に効く可能性）

測るもの: 各脚の単体性能 / L との誤差相関 / L に少量ブレンドしたときの対比較 t
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv3, legs
from config import PARAMS as _P
P = {**_P, "num_threads": 8}
ST, SEEDS = pd.Timestamp("2015-01-01"), [42, 1, 2]
WS = [0.05, 0.10, 0.15, 0.20, 0.30]

df = F.build()
rows = []
for fname, vs, ve in cv3.FOLDS:
    t0 = time.time()
    m_va = (df.date >= vs) & (df.date <= ve) & df.sales.notna()
    if m_va.sum() == 0: continue
    m_tr = (df.date >= ST) & (df.date < vs) & df.sales.notna()
    tr, va = df[m_tr], df[m_va]
    yv = va["y"].values

    ps = []
    for sd in SEEDS:
        mdl = lgb.train({**P, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd},
                        lgb.Dataset(tr[F.FEATURES], tr["y"], categorical_feature=F.CATS),
                        num_boost_round=500)
        ps.append(np.clip(mdl.predict(va[F.FEATURES]), 0, None))
    L = np.mean(ps, axis=0)

    P_ = {"L": L, "R": legs.ridge_leg(tr, va), "D": legs.dow_mean_leg(va),
          "N": legs.seasonal_naive_leg(va), "C": legs.croston_leg(tr, va)}
    row = {"fold": fname}
    rL = L - yv
    for k, v in P_.items():
        row[f"rmsle_{k}"] = float(np.sqrt(np.mean((v - yv) ** 2)))
        if k != "L":
            row[f"corr_{k}"] = float(np.corrcoef(rL, v - yv)[0, 1])   # 残差相関
            for w in WS:
                b = (1 - w) * L + w * v
                row[f"blend_{k}_{w}"] = float(np.sqrt(np.mean((np.clip(b, 0, None) - yv) ** 2)))
    rows.append(row)
    print(f"{fname}  L {row['rmsle_L']:.4f}  R {row['rmsle_R']:.4f}  D {row['rmsle_D']:.4f}  "
          f"N {row['rmsle_N']:.4f}  C {row['rmsle_C']:.4f}  |  resid corr R {row['corr_R']:.3f} C {row['corr_C']:.3f}  {time.time()-t0:.0f}s", flush=True)

R = pd.DataFrame(rows).set_index("fold")
R.to_csv(c.OUTPUT / "e31_hetero_legs.csv")
print(f"\n== {len(R)} fold 平均: 単体 RMSLE ==")
print(R[[f"rmsle_{k}" for k in "LRDNC"]].mean().round(4).to_string())
print("\n== L の残差との相関（1.0 に近いほど冗長、低いほど直交）==")
print(R[[f"corr_{k}" for k in "RDNC"]].mean().round(3).to_string())
print("\n== L 単体との対比較（採用は |t|>=3）==")
best = None
for k in "RDNC":
    for w in WS:
        d = (R[f"blend_{k}_{w}"] - R["rmsle_L"]).dropna()
        se = d.std() / np.sqrt(len(d)); t = d.mean() / se
        star = "★採用可" if t <= -3 else ""
        if star: best = (k, w, d.mean(), t)
        print(f"  脚{k} w={w:.2f}  平均差 {d.mean():+.4f}  勝率 {100*(d<0).mean():3.0f}%  t {t:+.2f}  {star}")
print(f"\n最良: {best}" if best else "\n|t|>=3 を満たすブレンドなし")
