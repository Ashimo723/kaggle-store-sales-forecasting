"""features2: 特徴量の大幅拡張。features.py の56特徴に以下を追加する。

欠落していたもの:
 A. transactions（店舗日次客数）— 一度も使っていなかった。2017-08-15 までだが日付基準 lag>=16 なら test 全日で取得可
 B. 集約系列（store合計 / family合計 / cluster×family）のラグ — 他系列から情報を借りる経路が皆無だった
 C. 未来側の onpromotion — test 期間の onpromotion は与えられているので前後窓・未来窓が合法
 D. 系列の状態量 — ゼロ率 / 最後に売れてからの日数 / rolling median,max,min / EWM
 E. カレンダー — 祝日までの日数・祝日からの日数 / 月内週 / 地震(2016-04-16)

ホライズン整合の原則は維持: 売上・transactions 由来はすべて日付基準 lag>=16。
onpromotion / holidays / oil のみ test でも既知なので未来側を使う。
"""
import numpy as np
import pandas as pd
import common as c
import features as F

EQ_DATE = pd.Timestamp("2016-04-16")  # マグニチュード7.8の地震


def _lagged_rolls(s, gkey, lags, rolls, prefix):
    """log1p 系列 s を gkey ごとに lag16 起点で集約する。返り値: dict[name] = Series
    gkey は配列、または複数キーなら配列のリスト（pandas に列名と解釈されないよう必ず値で渡す）。"""
    out = {}
    g = s.groupby(gkey, observed=True)
    for l in lags:
        out[f"{prefix}lag{l}"] = g.shift(l).astype("float32")
    base = g.shift(16)
    gb = base.groupby(gkey, observed=True)
    for w in rolls:
        out[f"{prefix}rmean{w}"] = gb.transform(lambda x, w=w: x.rolling(w, min_periods=1).mean()).astype("float32")
    return out


def add_transactions(df, raw):
    """A. 店舗日次客数。store×date の系列なので store 単位で作ってから join する。"""
    tx = raw["transactions"].copy()
    stores = df["store_nbr"].astype(int).unique()
    dates = pd.date_range(df.date.min(), df.date.max())
    grid = pd.MultiIndex.from_product([sorted(stores), dates], names=["store_nbr", "date"])
    t = (tx.assign(store_nbr=tx.store_nbr.astype(int)).set_index(["store_nbr", "date"])["transactions"]
         .reindex(grid))
    t = np.log1p(t).to_frame("txlog").reset_index().sort_values(["store_nbr", "date"])
    s = t.set_index(["store_nbr", "date"])["txlog"]
    d = _lagged_rolls(s, t["store_nbr"].values, [16, 21, 28], [7, 28, 112], "tx_")
    for k, v in d.items():
        t[k] = v.values
    cols = ["store_nbr", "date"] + list(d)
    return df.merge(t[cols], on=["store_nbr", "date"], how="left")


def add_aggregates(df):
    """B. 集約系列のラグ。store合計 / family合計 / cluster×family平均。
    log1p の平均を取る（指標空間と一致させる）。"""
    out = df
    specs = [
        (["store_nbr", "date"], "store_nbr", "agS_"),        # 店舗全体の売上水準
        (["family", "date"], "family", "agF_"),              # family 全体の需要
        (["cluster", "family", "date"], ["cluster", "family"], "agCF_"),  # 似た店舗群での同 family
    ]
    for keys, gk, pre in specs:
        a = df.groupby(keys, observed=True)["y"].mean().rename("v").reset_index()
        a = a.sort_values(keys)
        gkey = a[gk].values if isinstance(gk, str) else [a[col].values for col in gk]
        d = _lagged_rolls(a.set_index(keys)["v"], gkey, [16, 21], [7, 28], pre)
        for k, v in d.items():
            a[k] = v.values
        out = out.merge(a[keys + list(d)], on=keys, how="left")
    return out


def add_future_promo(df):
    """C. 未来側の onpromotion（test 期間も与えられているので合法）。"""
    g = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"]
    # 予測日を中心とした窓 / 未来7日 / 前日比
    df["promo_ctr7"] = g.transform(lambda s: s.rolling(7, center=True, min_periods=1).mean()).astype("float32")
    df["promo_ctr15"] = g.transform(lambda s: s.rolling(15, center=True, min_periods=1).mean()).astype("float32")
    df["promo_fwd7"] = g.transform(lambda s: s[::-1].rolling(7, min_periods=1).mean()[::-1]).astype("float32")
    df["promo_lead1"] = g.shift(-1).astype("float32")
    df["promo_store_fwd"] = df.groupby(["store_nbr", "date"], observed=True)["promo_ctr7"].transform("sum").astype("float32")
    # 系列の通常プロモ水準からの乖離（プロモの「珍しさ」）
    df["promo_dev"] = (np.log1p(df["onpromotion"]) - np.log1p(df["promo_ma28"])).astype("float32")
    return df


def add_state(df):
    """D. 系列の状態量。すべて lag16 起点。"""
    g = df.groupby(["store_nbr", "family"], observed=True)["y"]
    base = g.shift(16)
    gb = base.groupby([df.store_nbr, df.family], observed=True)
    for w in [28, 91]:
        df[f"zero_rate{w}"] = gb.transform(
            lambda x, w=w: (x <= 1e-9).rolling(w, min_periods=1).mean()).astype("float32")
    for w in [28, 91]:
        df[f"rmed{w}"] = gb.transform(lambda x, w=w: x.rolling(w, min_periods=1).median()).astype("float32")
    df["rmax91"] = gb.transform(lambda x: x.rolling(91, min_periods=1).max()).astype("float32")
    df["rmin28"] = gb.transform(lambda x: x.rolling(28, min_periods=1).min()).astype("float32")
    for a in [0.1, 0.3]:
        df[f"ewm{a}"] = gb.transform(lambda x, a=a: x.ewm(alpha=a, min_periods=1).mean()).astype("float32")
    # 最後に売れてからの連続ゼロ日数（lag16 時点）
    def _days_since(x):
        nz = x > 1e-9
        return (~nz).groupby(nz.cumsum()).cumsum()
    df["days_since_sale"] = gb.transform(_days_since).astype("float32")
    return df


def add_calendar2(df, raw):
    """E. 祝日までの距離・月内週・地震。"""
    ho = raw["holidays"]
    nat = pd.to_datetime(sorted(ho[(ho.locale == "National") & (~ho.transferred) &
                                   (ho.type.isin(["Holiday", "Additional", "Transfer", "Bridge"]))].date.unique()))
    dates = pd.date_range(df.date.min(), df.date.max())
    nxt = pd.Series(np.searchsorted(nat, dates), index=dates)
    d_next = pd.Series([(nat[min(i, len(nat) - 1)] - d).days for i, d in zip(nxt.values, dates)], index=dates)
    prv = np.clip(nxt.values - 1, 0, None)
    d_prev = pd.Series([(d - nat[i]).days for i, d in zip(prv, dates)], index=dates)
    m = pd.DataFrame({"days_to_hol": d_next.clip(0, 60).values,
                      "days_from_hol": d_prev.clip(0, 60).values}, index=dates)
    df = df.merge(m, left_on="date", right_index=True, how="left")
    df["week_of_month"] = ((df["day"] - 1) // 7).astype("int8")
    df["is_month_end3"] = (df["day"] > df["days_in_month"] - 3).astype("int8")
    eq = (df["date"] - EQ_DATE).dt.days
    df["eq_days"] = eq.where((eq >= 0) & (eq <= 120), -1).astype("int16")
    return df


def build(cache=True):
    path = c.ROOT / "output" / "panel2.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    df = F.build()
    raw = c.load_raw()
    df = add_transactions(df, raw)
    df = add_aggregates(df)
    df = add_future_promo(df)
    df = add_state(df)
    df = add_calendar2(df, raw)
    df.to_parquet(path, index=False)
    return df


NEW = (["tx_lag16", "tx_lag21", "tx_lag28", "tx_rmean7", "tx_rmean28", "tx_rmean112"]
       + [f"{p}{n}" for p in ["agS_", "agF_", "agCF_"] for n in ["lag16", "lag21", "rmean7", "rmean28"]]
       + ["promo_ctr7", "promo_ctr15", "promo_fwd7", "promo_lead1", "promo_store_fwd", "promo_dev"]
       + ["zero_rate28", "zero_rate91", "rmed28", "rmed91", "rmax91", "rmin28", "ewm0.1", "ewm0.3", "days_since_sale"]
       + ["days_to_hol", "days_from_hol", "week_of_month", "is_month_end3", "eq_days"])
FEATURES = F.FEATURES + NEW
CATS = F.CATS

if __name__ == "__main__":
    import time
    t0 = time.time()
    d = build(cache=False)
    print(f"shape {d.shape}  {time.time()-t0:.0f}s")
    print(f"特徴数 {len(F.FEATURES)} → {len(FEATURES)} (+{len(NEW)})")
    te = d[d.is_test]
    na = te[NEW].isna().mean()
    print("\n== test 行で NaN が残る新特徴（ホライズン整合の検算）==")
    print(na[na > 0].round(4).to_string() if (na > 0).any() else "  なし（全新特徴が test で取得可能）")
