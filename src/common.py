"""共通ユーティリティ: データロード / RMSLE / ローリング16日CV分割 / 提出書き出し。

設計上の約束（README.md「CV設計の要点」と対応）:
- 予測ホライズン H=16 日。fold は必ず 16 日連続ブロック。
- 学習に使ってよいのは fold 開始日より前のデータのみ（境界は inclusive/exclusive を明示）。
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
H = 16  # 予測ホライズン（test = 2017-08-16..08-31）
TEST_START = pd.Timestamp("2017-08-16")


def load_raw():
    """生データを dict で返す。date 列は datetime64。"""
    d = {}
    d["train"] = pd.read_csv(DATA / "train.csv", parse_dates=["date"])
    d["test"] = pd.read_csv(DATA / "test.csv", parse_dates=["date"])
    d["stores"] = pd.read_csv(DATA / "stores.csv")
    d["holidays"] = pd.read_csv(DATA / "holidays_events.csv", parse_dates=["date"])
    d["oil"] = pd.read_csv(DATA / "oil.csv", parse_dates=["date"])
    d["transactions"] = pd.read_csv(DATA / "transactions.csv", parse_dates=["date"])
    return d


def rmsle(y_true, y_pred):
    """RMSLE。予測は 0 未満を 0 にクリップしてから評価する（提出時も同じ扱い）。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


def rmsle_by(df, group_cols, y_col="sales", p_col="pred"):
    """グループ別 RMSLE。pooled 平均が大きい系列に支配されるのを見抜くため。"""
    return (
        df.groupby(group_cols, observed=True)
        .apply(lambda g: rmsle(g[y_col], g[p_col]), include_groups=False)
        .rename("rmsle")
    )


def make_folds(n_folds=5, last_valid_end=pd.Timestamp("2017-08-15"), horizon=H, gap_days=0):
    """末尾から horizon 日ブロックを n_folds 個、過去に向かって切り出す。

    返り値: [(train_end_exclusive, valid_start, valid_end), ...] 新しい fold が先頭。
    train は date < train_end_exclusive のみ使用可（train_end_exclusive = valid_start - gap_days）。
    """
    folds = []
    end = pd.Timestamp(last_valid_end)
    for _ in range(n_folds):
        start = end - pd.Timedelta(days=horizon - 1)
        folds.append((start - pd.Timedelta(days=gap_days), start, end))
        end = start - pd.Timedelta(days=1)
    return folds


def make_submission(test_df, pred, name, message=None):
    """id,sales の提出CSVを output/ に書く。負値クリップと行数・id 整合を検査。"""
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)
    assert len(pred) == len(test_df) == 28512, f"行数不正: {len(pred)} vs {len(test_df)}"
    sub = pd.DataFrame({"id": test_df["id"].values, "sales": pred})
    ref = pd.read_csv(DATA / "sample_submission.csv", usecols=["id"])
    assert sub["id"].tolist() == ref["id"].tolist(), "id の順序/内容が sample_submission と不一致"
    assert sub["sales"].notna().all(), "NaN が含まれる"
    path = OUTPUT / f"{name}.csv"
    sub.to_csv(path, index=False)
    print(f"wrote {path}  mean={sub.sales.mean():.3f} zero_ratio={(sub.sales == 0).mean():.3f}")
    if message:
        print(f"提出コマンド（要ユーザー承認）:\n  kaggle competitions submit -c store-sales-time-series-forecasting -f {path} -m \"{message}\"")
    return path
