"""e10: 後処理「直近N日の売上が全ゼロの系列は 0 と予測する」（i11）。
学習不要 — e02 の OOF 予測に対して後処理を当てるだけなので CPU をほぼ使わない。
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

oof = pd.read_parquet(c.OUTPUT / "e02_oof.parquet")
tr = pd.read_csv(c.DATA / "train.csv", parse_dates=["date"])
folds = c.make_folds(5)
base = float(np.sqrt(np.mean((oof.pred - oof.y) ** 2)))
print(f"e02 OOF ベース RMSLE {base:.4f}\n")

print("== 直近N日が全ゼロの系列を強制ゼロ化 ==")
print("N     対象系列数  対象行の割合   RMSLE      差")
for N in [30, 60, 90, 120, 180, 365]:
    p = oof.pred.values.copy()
    hit_series, hit_rows = 0, 0
    for k, (tr_end, vs, ve) in enumerate(folds):
        h = tr[(tr.date < tr_end) & (tr.date >= tr_end - pd.Timedelta(days=N))]
        dead = h.groupby(["store_nbr", "family"], observed=True).sales.sum()
        dead = set(dead[dead == 0].index)
        m = (oof.fold.values == k) & np.array([(s, f) in dead for s, f in zip(oof.store_nbr, oof.family)])
        p[m] = 0.0
        hit_series += len(dead); hit_rows += m.sum()
    s = float(np.sqrt(np.mean((p - oof.y.values) ** 2)))
    print(f"{N:<5d} {hit_series/5:9.1f} {hit_rows/len(oof)*100:11.2f}%  {s:.4f}  {s-base:+.4f}")

print("\n== 参考: そもそも e02 は死んだ系列に何を予測しているか（N=60 の対象行）==")
k0 = folds[0]
h = tr[(tr.date < k0[0]) & (tr.date >= k0[0] - pd.Timedelta(days=60))]
dead = h.groupby(["store_nbr", "family"], observed=True).sales.sum()
dead = set(dead[dead == 0].index)
m = (oof.fold.values == 0) & np.array([(s, f) in dead for s, f in zip(oof.store_nbr, oof.family)])
sub = oof[m]
print(f"対象行 {len(sub)}   予測 y(log1p) の平均 {sub.pred.mean():.4f} / 最大 {sub.pred.max():.4f}")
print(f"真値が 0 でない行の割合: {(sub.y > 0).mean()*100:.2f}%   真値>0 の平均 {sub.loc[sub.y>0,'y'].mean() if (sub.y>0).any() else 0:.4f}")
print(f"この対象行だけの RMSLE: e02 {np.sqrt((sub.pred-sub.y).pow(2).mean()):.4f} → 強制0 {np.sqrt(sub.y.pow(2).mean()):.4f}")
