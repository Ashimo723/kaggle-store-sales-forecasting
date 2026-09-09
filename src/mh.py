"""direct multi-horizon（origin 基準）のデータ構築。

現行（日付基準）: 予測日 t に対し lag_k = y(t-k), k>=16。
  → h=1 の予測(8/16)にも16日前(7/31)の情報しか使えていない。
origin 基準: origin = 最終学習日。予測日 t = origin + h (h=1..16)。
  特徴は origin から見たラグ lag_k = y(origin-k), **k>=0** で、horizon h も特徴に入れる。
  → h=1 なら前日、h=8 なら8日前が使える。test でも 8/15 まで既知なので完全に合法。

同曜日の扱い: 予測日 t と同曜日で origin 以前の直近は t - 7*ceil(h/7)。
  origin からの距離は d0 = 7*ceil(h/7) - h（0..6）。そこから 7 日刻みで4回分を取る。
"""
import numpy as np, pandas as pd
import common as c, features as F

OLAGS = list(range(0, 22)) + [28, 35, 42, 49, 56, 63, 91, 182, 364]   # origin 基準のラグ
OROLL = [7, 14, 28, 56, 112, 365]
CAL = ["dow", "day", "month", "year", "dayofyear", "weekofyear", "is_weekend", "is_payday",
       "days_since_payday", "days_to_payday", "is_nye", "hol_national", "hol_regional",
       "hol_local", "hol_any", "is_event", "is_workday", "dcoilwtico", "oil_ma28",
       "onpromotion", "promo_ma7", "promo_ma28", "promo_store_day"]
STATIC = ["store_nbr", "family", "city", "state", "type", "cluster"]
H = 16


def build_origin_panel():
    """origin 側の特徴（lag0.. と rolling）を持つパネル。日付ごとに『その日を origin としたときの特徴』。"""
    df = F.build()[["store_nbr", "family", "date", "y", "sales", "is_test"]].copy()
    df = df.sort_values(["store_nbr", "family", "date"], ignore_index=True)
    g = df.groupby(["store_nbr", "family"], observed=True)["y"]
    out = {"store_nbr": df.store_nbr, "family": df.family, "date": df.date}
    for l in OLAGS:
        out[f"o{l}"] = (df["y"] if l == 0 else g.shift(l)).astype("float32")
    base = df["y"]
    gb = base.groupby([df.store_nbr, df.family], observed=True)
    for w in OROLL:
        out[f"orm{w}"] = gb.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean()).astype("float32")
    for w in [28, 112]:
        out[f"ors{w}"] = gb.transform(lambda x, w=w: x.rolling(w, min_periods=2).std()).astype("float32")
    return pd.DataFrame(out)


def make_samples(og, target, origins):
    """origins（日付の配列）から h=1..16 のサンプルを作る。
    og: build_origin_panel() の出力 / target: 予測日側の情報（カレンダー・promo・y）"""
    o = og[og.date.isin(origins)]
    frames = []
    for h in range(1, H + 1):
        t = o.copy()
        t["h"] = np.int8(h)
        t["date"] = t["date"] + pd.Timedelta(days=h)
        frames.append(t)
    X = pd.concat(frames, ignore_index=True)
    X = X.merge(target, on=["store_nbr", "family", "date"], how="inner")
    # 予測日と同曜日で origin 以前の直近4回（origin からの距離 d0 + 7j）
    d0 = (7 * np.ceil(X["h"].values / 7) - X["h"].values).astype(int)
    lagmat = X[[f"o{l}" for l in range(0, 22)]].values
    for j in range(4):
        idx = d0 + 7 * j
        ok = idx < 22
        v = np.full(len(X), np.nan, dtype="float32")
        v[ok] = lagmat[np.arange(len(X))[ok], idx[ok]]
        X[f"dowr{j+1}"] = v
    X["dow_mean4"] = X[[f"dowr{j+1}" for j in range(4)]].mean(axis=1).astype("float32")
    return X


FEATURES = (STATIC + CAL + ["h"] + [f"o{l}" for l in OLAGS] + [f"orm{w}" for w in OROLL]
            + ["ors28", "ors112"] + [f"dowr{j+1}" for j in range(4)] + ["dow_mean4"])
CATS = STATIC
