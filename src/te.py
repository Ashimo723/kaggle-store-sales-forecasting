"""ターゲットエンコーディング。時系列リークを避けるため fold の train 期間のみで統計を作る。

設計:
- 生の y ではなく **残差 resid = y - (store,family) の直近水準** をエンコードする。
  生 y をエンコードすると rmean7/rmean14（gain の84%）と同じ水準情報の重複になり、
  木が既に持っている情報を増やすだけで新規性が無い。残差なら「曜日・日・プロモへの反応の形」が入る。
- 件数の少ないキーは全体平均へ縮小（smoothing m）。
"""
import numpy as np
import pandas as pd

# (名前, キー列) 。水準を除いた残差に対する平均をエンコードする
TE_KEYS = [
    ("te_sf_dow",      ["store_nbr", "family", "dow"]),        # 系列ごとの曜日パターン
    ("te_f_dow",       ["family", "dow"]),                     # family 共通の曜日パターン
    ("te_cl_f_dow",    ["cluster", "family", "dow"]),          # 店舗クラスタ x family の曜日
    ("te_sf_day",      ["store_nbr", "family", "day"]),        # 給与日を含む月内パターン
    ("te_f_day",       ["family", "day"]),
    ("te_sf_promo",    ["store_nbr", "family", "promo_bin"]),  # プロモ感度（系列別）
    ("te_f_promo",     ["family", "promo_bin"]),
    ("te_f_month",     ["family", "month"]),                   # 年次季節性（SCHOOL 等）
    ("te_sf_hol",      ["store_nbr", "family", "hol_any"]),    # 祝日反応（系列別）
    ("te_f_doy_bin",   ["family", "doy_bin"]),                 # 年内位置（10日刻み）
]


def _prep(df):
    d = pd.DataFrame(index=df.index)
    d["promo_bin"] = np.minimum(df["onpromotion"].values, 3).astype("int8")  # 0,1,2,3+
    d["doy_bin"] = (df["dayofyear"].values // 10).astype("int16")
    return d


def add_te(df, train_mask, level_days=365, m=20, keys=TE_KEYS, expanding=True):
    """train_mask=True の行だけで統計を作り、df 全体に付与して返す（新規列名のリスト付き）。

    level_days: 系列水準を計算する期間（train 期間の末尾 N 日）。長すぎるとドリフトを拾う。
    expanding: True なら学習行の TE を「その行より前の train データのみ」で作る（自己リーク回避）。
      False（e04で測定）だと学習行が自分自身を含む統計を見るため、train で過剰に効いて即 early stop する。
      valid/test 行には train 全体の統計を使う（実配備と同じ条件）。
    """
    work = df[["store_nbr", "family", "cluster", "dow", "day", "month",
               "dayofyear", "hol_any", "onpromotion", "y", "date"]].copy()
    work = pd.concat([work, _prep(df)], axis=1)

    tr = work[train_mask]
    lvl_from = tr["date"].max() - pd.Timedelta(days=level_days)
    lvl = (tr[tr.date >= lvl_from].groupby(["store_nbr", "family"], observed=True)["y"]
           .mean().rename("lvl"))
    work = work.join(lvl, on=["store_nbr", "family"])
    work["lvl"] = work["lvl"].fillna(lvl.mean())
    work["resid"] = work["y"] - work["lvl"]

    tr = work[train_mask]
    gmean = tr["resid"].mean()
    new_cols = []
    order = tr.sort_values("date").index  # 時系列順（expanding 用）
    for name, key in keys:
        # valid/test 行用: train 全体の統計（実配備と同じ）
        agg = tr.groupby(key, observed=True)["resid"].agg(["mean", "count"])
        sm = ((agg["mean"] * agg["count"] + gmean * m) / (agg["count"] + m)).rename(name)
        work = work.join(sm, on=key)
        work[name] = work[name].fillna(gmean).astype("float32")
        if expanding:
            # 学習行用: その行より前の train データのみで再計算して上書き
            t = tr.loc[order, key + ["resid"]]
            g = t.groupby(key, observed=True)["resid"]
            csum = g.cumsum() - t["resid"]   # 自分を除く累積和
            ccnt = g.cumcount()              # 自分より前の件数
            work.loc[order, name] = (((csum + gmean * m) / (ccnt + m))
                                     .astype("float32").values)
        new_cols.append(name)
    # 系列水準そのものも特徴として渡す（rmean と別期間の情報）
    work["te_level"] = work["lvl"].astype("float32")
    new_cols.append("te_level")
    return work[new_cols], new_cols
