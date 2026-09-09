"""特徴量生成。ホライズン H=16 日に対して整合する特徴だけを作る。

重要な約束:
- ラグは **日付基準で lag >= 16 のみ**。test(8/16-8/31)の予測時、最新の既知売上は 8/15 なので
  lag16 は 8/16 の予測にも 8/31 の予測にも実在する。lag<16 は test で入手不能＝使用禁止。
- transactions.csv は 2017-08-15 までしか無いので **特徴に使わない**（使うと配備不能）。
- onpromotion / oil / holidays は test 期間分が与えられているので当日値を使ってよい。
"""
import numpy as np
import pandas as pd
import common as c

LAGS = [16, 17, 18, 19, 20, 21, 28, 35, 42, 49, 56, 63, 91, 182, 364]
ROLL = [7, 14, 28, 56, 112, 365]  # lag16 を起点とする移動集約の窓


def build_panel():
    """train+test を「全系列 x 全日付」の完全グリッドに整形する。
    12/25 の欠測4日は sales=NaN の行として明示的に埋める（shift のずれを防ぐ）。"""
    raw = c.load_raw()
    tr, te = raw["train"], raw["test"]
    tr["is_test"] = False
    te["is_test"] = True
    df = pd.concat([tr, te], ignore_index=True)

    dates = pd.date_range(df.date.min(), df.date.max())
    keys = df[["store_nbr", "family"]].drop_duplicates()
    grid = keys.merge(pd.DataFrame({"date": dates}), how="cross")
    df = grid.merge(df, on=["store_nbr", "family", "date"], how="left")
    df["is_test"] = df["is_test"].fillna(False)
    df["onpromotion"] = df["onpromotion"].fillna(0)
    df = df.sort_values(["store_nbr", "family", "date"], ignore_index=True)
    return df, raw


def add_calendar(df):
    d = df["date"].dt
    df["dow"] = d.dayofweek.astype("int8")
    df["day"] = d.day.astype("int8")
    df["month"] = d.month.astype("int8")
    df["year"] = d.year.astype("int16")
    df["dayofyear"] = d.dayofyear.astype("int16")
    df["weekofyear"] = d.isocalendar().week.values.astype("int16")
    df["days_in_month"] = d.days_in_month.astype("int8")
    df["is_weekend"] = (df["dow"] >= 5).astype("int8")
    # 給与日: 15日 と 月末（公務員の月2回払い）。直近の給与日からの経過日数
    df["is_payday"] = ((df["day"] == 15) | (df["day"] == df["days_in_month"])).astype("int8")
    df["days_since_payday"] = np.where(df["day"] >= 15, df["day"] - 15, df["day"]).astype("int8")
    df["days_to_payday"] = np.where(
        df["day"] < 15, 15 - df["day"], df["days_in_month"] - df["day"]
    ).astype("int8")
    df["is_nye"] = ((df["month"] == 1) & (df["day"] == 1)).astype("int8")  # 1/1 はほぼ全店休業
    return df


def add_holidays(df, raw):
    """locale を store の city/state と突き合わせる。全店一律フラグにしない（i05）。"""
    ho = raw["holidays"].copy()
    ho = ho[~((ho.type == "Holiday") & (ho.transferred))]  # 移動元は祝日でない
    real = ho[ho.type.isin(["Holiday", "Additional", "Transfer", "Bridge"])]
    nat = real[real.locale == "National"].groupby("date").size().rename("hol_national")
    reg = real[real.locale == "Regional"].groupby(["date", "locale_name"]).size().rename("hol_regional")
    loc = real[real.locale == "Local"].groupby(["date", "locale_name"]).size().rename("hol_local")
    ev = ho[ho.type == "Event"].groupby("date").size().rename("is_event")
    wd = ho[ho.type == "Work Day"].groupby("date").size().rename("is_workday")

    df = df.merge(nat, left_on="date", right_index=True, how="left")
    df = df.merge(reg, left_on=["date", "state"], right_index=True, how="left")
    df = df.merge(loc, left_on=["date", "city"], right_index=True, how="left")
    df = df.merge(ev, left_on="date", right_index=True, how="left")
    df = df.merge(wd, left_on="date", right_index=True, how="left")
    for col in ["hol_national", "hol_regional", "hol_local", "is_event", "is_workday"]:
        df[col] = df[col].fillna(0).clip(0, 1).astype("int8")
    df["hol_any"] = df[["hol_national", "hol_regional", "hol_local"]].max(axis=1).astype("int8")
    return df


def add_oil(df, raw):
    oil = raw["oil"].set_index("date").reindex(pd.date_range("2013-01-01", "2017-08-31"))
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()  # 週末・祝日の欠損を前方補完
    oil["oil_ma28"] = oil["dcoilwtico"].rolling(28, min_periods=1).mean()
    return df.merge(oil, left_on="date", right_index=True, how="left")


def add_lags(df):
    """log1p 空間でラグ・移動集約を作る（目的が log1p なので同じ空間で持つ）。"""
    df["y"] = np.log1p(df["sales"])
    g = df.groupby(["store_nbr", "family"], observed=True)["y"]
    for l in LAGS:
        df[f"lag{l}"] = g.shift(l).astype("float32")
    base = g.shift(16)  # 移動集約はすべて lag16 を起点にする＝ホライズン整合
    gb = base.groupby([df.store_nbr, df.family], observed=True)
    for w in ROLL:
        df[f"rmean{w}"] = gb.transform(lambda s, w=w: s.rolling(w, min_periods=1).mean()).astype("float32")
    for w in [28, 112]:
        df[f"rstd{w}"] = gb.transform(lambda s, w=w: s.rolling(w, min_periods=2).std()).astype("float32")
    # 同曜日の直近平均 = e01 best ベースラインのホライズン整合版。
    # 予測日 t と同曜日で lag>=16 を満たすのは t-21, t-28, t-35, t-42（t-7,t-14 は test で入手不能）
    df["dow4_mean"] = df[["lag21", "lag28", "lag35", "lag42"]].mean(axis=1).astype("float32")
    df["dow2_mean"] = df[["lag21", "lag28"]].mean(axis=1).astype("float32")
    # トレンド: 直近28日平均と その前の28日平均の差
    df["trend_28"] = (df["rmean28"] - (df["rmean56"] * 2 - df["rmean28"])).astype("float32")
    # 前年同期比（水準でなく比。SCHOOL AND OFFICE SUPPLIES の年次季節性向け i17）
    df["yoy_ratio"] = (df["lag364"] - df["rmean365"]).astype("float32")
    return df


def add_promo(df):
    g = df.groupby(["store_nbr", "family"], observed=True)["onpromotion"]
    # onpromotion は test 期間も既知なので、未来側の窓も使ってよい
    df["promo_ma7"] = g.transform(lambda s: s.rolling(7, min_periods=1).mean()).astype("float32")
    df["promo_ma28"] = g.transform(lambda s: s.rolling(28, min_periods=1).mean()).astype("float32")
    df["promo_store_day"] = df.groupby(["store_nbr", "date"], observed=True)["onpromotion"].transform("sum").astype("float32")
    df["onpromotion"] = df["onpromotion"].astype("float32")
    return df


def build(cache=True):
    path = c.ROOT / "output" / "panel.parquet"
    if cache and path.exists():
        return pd.read_parquet(path)
    df, raw = build_panel()
    df = df.merge(raw["stores"], on="store_nbr", how="left")
    df = add_calendar(df)
    df = add_holidays(df, raw)
    df = add_oil(df, raw)
    df = add_promo(df)
    df = add_lags(df)
    for col in ["family", "city", "state", "type"]:
        df[col] = df[col].astype("category")
    df["store_nbr"] = df["store_nbr"].astype("category")
    df["cluster"] = df["cluster"].astype("category")
    df.to_parquet(path, index=False)
    return df


FEATURES = (
    ["store_nbr", "family", "city", "state", "type", "cluster"]
    + ["dow", "day", "month", "year", "dayofyear", "weekofyear", "is_weekend",
       "is_payday", "days_since_payday", "days_to_payday", "is_nye"]
    + ["hol_national", "hol_regional", "hol_local", "hol_any", "is_event", "is_workday"]
    + ["dcoilwtico", "oil_ma28"]
    + ["onpromotion", "promo_ma7", "promo_ma28", "promo_store_day"]
    + [f"lag{l}" for l in LAGS]
    + [f"rmean{w}" for w in ROLL] + ["rstd28", "rstd112", "dow4_mean", "dow2_mean", "trend_28", "yoy_ratio"]
)
CATS = ["store_nbr", "family", "city", "state", "type", "cluster"]

if __name__ == "__main__":
    d = build(cache=False)
    print(d.shape)
    print("欠測日を埋めた行数(sales NaN かつ非test):", int(d.sales.isna().sum() - d.is_test.sum()))
    print("特徴数:", len(FEATURES))
    print(d[d.is_test][["date", "lag16", "lag364", "rmean28", "dow4_mean"]].isna().mean().to_string())
