"""祝日を「種類ごとの個別ダミー」に展開する（公開ノート LB 0.37984 からの知見）。

我々は hol_national/hol_regional/hol_local の 0/1 フラグ3本に集約していたが、
祝日は種類ごとに効果が正反対（クリスマスは売上急増、元日はほぼ全店休業）。
description を正規化してダミー化し、捨てていた情報を復元する。

正規化: 地名を除去 / "+1","-1" などの前後日サフィックスを除去 / traslado・puente・recupero を除去
        （"futbol" だけは地名除去の対象外にする）
"""
import re
import numpy as np, pandas as pd
import common as c

# 公開ノートが「売上への影響が大きい」として選んだ全国祝日
SELECTED_NATIONAL = ["terremoto", "navidad", "dia la madre", "dia trabajo",
                     "primer dia ano", "futbol", "dia difuntos"]


def normalize(hol, stores):
    """description を正規化する"""
    places = set(stores.city.str.lower()) | set(stores.state.str.lower())
    def f(row):
        s = str(row["description"]).lower().replace(str(row["locale_name"]).lower(), "")
        if "futbol" in s:
            return "futbol"
        for w in places:
            s = s.replace(w, "")
        return s
    d = hol.apply(f, axis=1)
    d = d.replace(r"[+-]\d+|\b(de|del|traslado|recupero|puente|-)\b", "", regex=True)
    d = d.replace(r"\s+|-", " ", regex=True).str.strip()
    return d


def build(selected=None):
    """returns: (work_days, national_dummies, local_dummies, regional_dummies)"""
    raw = c.load_raw()
    hol, stores = raw["holidays"].copy(), raw["stores"]
    hol["description"] = normalize(hol, stores)
    hol = hol[~hol.transferred]                      # 移動元は祝日でない

    work = hol[hol.type == "Work Day"][["date"]].assign(work_day=1).drop_duplicates()
    hol = hol[hol.type != "Work Day"]

    nat = hol[hol.locale == "National"][["date", "description"]].drop_duplicates()
    nat = pd.get_dummies(nat, columns=["description"], prefix="nat").groupby("date").max().reset_index()
    keep = [f"nat_{s}" for s in (selected or SELECTED_NATIONAL) if f"nat_{s}" in nat.columns]
    nat = nat[["date"] + keep]

    loc = hol[hol.locale == "Local"][["date", "locale_name", "description"]].drop_duplicates()
    loc = pd.get_dummies(loc.rename(columns={"locale_name": "city"}), columns=["description"], prefix="loc")
    loc = loc.groupby(["date", "city"]).max().reset_index()

    reg = hol[hol.locale == "Regional"][["date", "locale_name"]].drop_duplicates()
    reg = reg.rename(columns={"locale_name": "state"}).assign(reg_prov=1)

    for d in (work, nat, loc, reg):
        d[[c_ for c_ in d.columns if c_ != "date" and d[c_].dtype == bool]] = \
            d[[c_ for c_ in d.columns if c_ != "date" and d[c_].dtype == bool]].astype("int8")
    return work, nat, loc, reg


if __name__ == "__main__":
    w, n, l, r = build()
    print("work days:", len(w))
    print("national dummies:", [x for x in n.columns if x != "date"])
    print("local dummy 列数:", len([x for x in l.columns if x.startswith("loc_")]), " 行数:", len(l))
    print("regional:", len(r))
    raw = c.load_raw(); h = raw["holidays"].copy()
    h["d"] = normalize(h, raw["stores"])
    print("\n=== 正規化後の description 頻度 上位20 ===")
    print(h[h.locale == "National"].d.value_counts().head(20).to_string())
