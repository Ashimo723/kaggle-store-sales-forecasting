"""e01: 古典ベースラインの比較（ローリング16日 x 5fold, RMSLE）。

RMSLE = log1p 空間の RMSE なので、集約はすべて log1p 空間で行う。
点予測の L2 最適は log1p 空間の「平均」（生売上空間では幾何平均寄り）。
比較のため中央値版も測る。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c

tr = pd.read_csv(c.DATA / "train.csv", parse_dates=["date"])
piv = tr.pivot_table(index="date", columns=["store_nbr", "family"], values="sales", aggfunc="mean")
piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max()))  # 12/25 の欠測日を NaN 行として明示
L = np.log1p(piv)  # (日数, 1782) log1p 空間
COLS = piv.columns


def _tail_agg(hist, ndays, how):
    """直近 ndays の集約（NaN無視）。hist: log1p の DataFrame"""
    w = hist.tail(ndays)
    return (w.mean() if how == "mean" else w.median()).fillna(0.0).values


def _dow_agg(hist, valid_dates, nweeks, how):
    """曜日別に直近 nweeks 週を集約し、valid 各日の曜日へ割り当てる"""
    tail = hist.tail(nweeks * 7)
    g = tail.groupby(tail.index.dayofweek)
    tbl = (g.mean() if how == "mean" else g.median())  # index=dow, cols=系列
    out = np.zeros((len(valid_dates), hist.shape[1]))
    for i, d in enumerate(valid_dates):
        dow = d.dayofweek
        out[i] = tbl.loc[dow].fillna(0.0).values if dow in tbl.index else 0.0
    return out


def predict(name, hist, valid_dates):
    n = len(valid_dates)
    if name == "zero":
        return np.zeros((n, hist.shape[1]))
    if name.startswith("last") and name.endswith("_mean"):
        return np.tile(_tail_agg(hist, int(name[4:-5]), "mean"), (n, 1))
    if name.startswith("last") and name.endswith("_median"):
        return np.tile(_tail_agg(hist, int(name[4:-7]), "median"), (n, 1))
    if name.startswith("dow") and name.endswith("_mean"):
        return _dow_agg(hist, valid_dates, int(name[3:-5]), "mean")
    if name.startswith("dow") and name.endswith("_median"):
        return _dow_agg(hist, valid_dates, int(name[3:-7]), "median")
    if name == "naive364":  # 前年同時期（曜日を保つ 52週前）
        idx = [d - pd.Timedelta(days=364) for d in valid_dates]
        return L.reindex(idx).fillna(0.0).values
    if name == "naive16":   # 16日前をそのままコピー（ホライズン整合の最短シフト）
        idx = [d - pd.Timedelta(days=16) for d in valid_dates]
        return L.reindex(idx).fillna(0.0).values
    raise ValueError(name)


METHODS = ["zero", "last16_mean", "last16_median", "last56_mean",
           "dow4_mean", "dow4_median", "dow8_mean", "dow8_median",
           "dow16_mean", "dow16_median", "naive16", "naive364"]

folds = c.make_folds(5)
rows = []
per_fold = {m: [] for m in METHODS}
for k, (train_end, vs, ve) in enumerate(folds):
    hist = L[L.index < train_end]
    vdates = pd.date_range(vs, ve)
    ytrue = L.reindex(vdates).values  # log1p 空間の真値
    assert not np.isnan(ytrue).any(), f"valid に欠測日 {vs}..{ve}"
    for m in METHODS:
        p = np.clip(predict(m, hist, vdates), 0, None)
        per_fold[m].append(float(np.sqrt(np.mean((p - ytrue) ** 2))))

print("== fold 別 RMSLE（fold0 = 最新 2017-07-31..08-15）==")
hdr = "method".ljust(15) + "".join(f"fold{k}".rjust(9) for k in range(5)) + "mean".rjust(9)
print(hdr); print("-" * len(hdr))
res = sorted(((np.mean(v), m, v) for m, v in per_fold.items()))
for mean, m, v in res:
    print(m.ljust(15) + "".join(f"{x:9.4f}" for x in v) + f"{mean:9.4f}")

best = res[0][1]
print(f"\nbest = {best}  (mean RMSLE {res[0][0]:.4f})")

# --- best 手法で test 予測を作る（提出はしない）---
hist_full = L[L.index <= pd.Timestamp("2017-08-15")]
vdates = pd.date_range(c.TEST_START, "2017-08-31")
pred_log = np.clip(predict(best, hist_full, vdates), 0, None)
pred_df = pd.DataFrame(np.expm1(pred_log), index=vdates, columns=COLS)
te = pd.read_csv(c.DATA / "test.csv", parse_dates=["date"])
te["pred"] = [pred_df.at[d, (s, f)] for d, s, f in zip(te.date, te.store_nbr, te.family)]
c.make_submission(te, te["pred"].values, f"e01_{best}", message=f"e01 baseline {best}")
