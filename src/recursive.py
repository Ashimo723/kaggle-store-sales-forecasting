"""再帰予測（1日先モデルを16回適用）。公開ノート LB 0.37984 の中核方式。

我々の direct 方式は「日付基準 lag>=16 で16日先を直接予測」。
公開ノートは output_chunk_length=1 / lags=63 で **1日先モデルを16回再帰適用**する。
  h=1 は前日(lag1)が使える。h>=2 は予測値を入力に積み上げる（誤差伝播あり）。

実装:
  学習 = lag1.. を含む特徴を groupby shift で一括生成（高速）
  予測 = (系列 x 日付) の行列を持ち、1日ずつ予測して書き戻す
"""
import numpy as np, pandas as pd, lightgbm as lgb
import common as c, features as F

LAGS = list(range(1, 22)) + [28, 35, 42, 49, 56, 63, 91, 182, 364]
ROLL = [7, 14, 28, 56, 112, 365]
CAL = ["dow", "day", "month", "year", "dayofyear", "weekofyear", "is_weekend", "is_payday",
       "days_since_payday", "days_to_payday", "is_nye", "hol_national", "hol_regional",
       "hol_local", "hol_any", "is_event", "is_workday", "dcoilwtico", "oil_ma28",
       "onpromotion", "promo_ma7", "promo_ma28", "promo_store_day"]
STATIC = ["store_nbr", "family", "city", "state", "type", "cluster"]
LAGF = [f"rl{l}" for l in LAGS] + [f"rm{w}" for w in ROLL] + ["rs28", "zr28"]
FEATURES = STATIC + CAL + LAGF
CATS = STATIC


def build(cache=True):
    """lag1 起点の特徴を持つパネル（再帰予測の学習用）"""
    path = c.ROOT / "output" / "panel_rec.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    df = F.build()
    g = df.groupby(["store_nbr", "family"], observed=True)["y"]
    for l in LAGS:
        df[f"rl{l}"] = g.shift(l).astype("float32")
    base = g.shift(1)                      # 1日先モデルなので lag1 起点
    gb = base.groupby([df.store_nbr, df.family], observed=True)
    for w in ROLL:
        df[f"rm{w}"] = gb.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean()).astype("float32")
    df["rs28"] = gb.transform(lambda x: x.rolling(28, min_periods=2).std()).astype("float32")
    df["zr28"] = gb.transform(lambda x: (x <= 1e-9).rolling(28, min_periods=1).mean()).astype("float32")
    keep = ["store_nbr", "family", "city", "state", "type", "cluster", "date", "y", "sales",
            "is_test", "id"] + CAL + LAGF
    df = df[[k for k in dict.fromkeys(keep) if k in df.columns]]
    df.to_parquet(path, index=False)
    return df


def _lag_block(M, pos):
    """行列 M（系列 x 日付、pos 列目が予測対象日）からラグ特徴を作る"""
    n = M.shape[0]
    out = {}
    for l in LAGS:
        j = pos - l
        out[f"rl{l}"] = M[:, j].copy() if j >= 0 else np.full(n, np.nan, dtype="float32")
    for w in ROLL:
        a = max(0, pos - w)
        out[f"rm{w}"] = (np.nanmean(M[:, a:pos], axis=1) if pos > a
                         else np.full(n, np.nan, dtype="float32")).astype("float32")
    a = max(0, pos - 28); win = M[:, a:pos]
    out["rs28"] = (np.nanstd(win, axis=1) if win.shape[1] > 1 else np.zeros(n)).astype("float32")
    out["zr28"] = ((win <= 1e-9).mean(axis=1) if win.shape[1] > 0 else np.zeros(n)).astype("float32")
    return out


def recursive_predict(models, df, keys, origin, horizon=16, zero_fc_window=0):
    """origin の翌日から horizon 日を再帰予測する。models は 1日先モデルのリスト（seed 平均）。

    zero_fc_window > 0 のとき、直近 N 日が全ゼロの系列は予測を 0 に固定する（公開ノートの zero_fc_window=21）。
    """
    piv = df.pivot_table(index=["store_nbr", "family"], columns="date", values="y",
                         aggfunc="mean", observed=True).reindex(keys)
    # 12/25 は train に行が無い欠測日。列が飛ぶと origin が引けないので日付を連続化する。
    # また pivot_table は全 NaN の列（= test 期間、y が未知）を落とすので df の日付範囲で補う
    piv = piv.reindex(columns=pd.date_range(df["date"].min(), df["date"].max()))
    M = piv.values.astype("float32")
    dates = list(piv.columns)
    d2p = {d: i for i, d in enumerate(dates)}
    opos = d2p[origin]
    M[:, opos + 1:] = np.nan                       # origin より後は未知にする

    dead = None
    if zero_fc_window > 0:
        w = M[:, max(0, opos + 1 - zero_fc_window):opos + 1]
        dead = np.nan_to_num(w, nan=0.0).sum(axis=1) <= 1e-9

    side = df.set_index(["store_nbr", "family", "date"])
    preds = {}
    for h in range(1, horizon + 1):
        d = origin + pd.Timedelta(days=h)
        if d not in d2p: break
        pos = d2p[d]
        rows = side.loc[(slice(None), slice(None), d), :].reset_index()
        rows = rows.set_index(["store_nbr", "family"]).reindex(keys).reset_index()
        X = rows[STATIC + CAL].copy()
        for col in STATIC:                      # reindex でカテゴリ dtype が壊れるので復元
            X[col] = X[col].astype(df[col].dtype)
        for k, v in _lag_block(M, pos).items():
            X[k] = v
        p = np.mean([np.clip(m.predict(X[FEATURES]), 0, None) for m in models], axis=0)
        if dead is not None:
            p[dead] = 0.0
        M[:, pos] = p.astype("float32")
        preds[d] = p
    return preds, keys
