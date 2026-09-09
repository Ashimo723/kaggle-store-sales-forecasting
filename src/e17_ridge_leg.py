"""e17: ラグを持たない脚（系列ごとの Ridge）。LightGBM とは情報源が異なる直交脚を作る。

動機: ここまでの施策は全て「単一 LightGBM の中の変更」で、ラグ・移動平均という同じ情報源に依存していた。
系列ごとに時間トレンド + 季節性 + プロモだけで回帰する脚を作れば、誤差が相関しにくい。
（この脚が入れば GRAVEYARD g03 の TE 再挑戦条件「ラグ特徴を持たない構成」も満たす）

特徴: 曜日ダミー / 月内位置 / 給与日 / 祝日 / onpromotion / 線形トレンド / dayofyear のフーリエ項
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F, cv2

K_FOURIER = 3  # 年次季節性のフーリエ次数


def design(d):
    """d: DataFrame（date, onpromotion, hol_any, is_payday, ...）→ 設計行列"""
    X = [np.ones(len(d))]
    dow = d["dow"].values
    for k in range(1, 7):                      # 曜日ダミー（月曜を基準）
        X.append((dow == k).astype(float))
    doy = d["dayofyear"].values / 365.25
    for k in range(1, K_FOURIER + 1):          # 年次季節性
        X.append(np.sin(2 * np.pi * k * doy)); X.append(np.cos(2 * np.pi * k * doy))
    X.append(d["is_payday"].values.astype(float))
    X.append(d["days_since_payday"].values / 15.0)
    X.append(d["hol_any"].values.astype(float))
    X.append(d["is_nye"].values.astype(float))
    X.append(np.log1p(d["onpromotion"].values))
    X.append((d["date"].values.astype("datetime64[D]").astype(int) - 16000) / 1000.0)  # 線形トレンド
    return np.column_stack(X)


df = F.build()
res = {}
for fl, vs, ve, drop in cv2.FOLDS:
    t0 = time.time()
    m_tr = (df.date >= pd.Timestamp("2015-01-01")) & (df.date < vs) & df.sales.notna()
    m_va = (df.date >= vs) & (df.date <= ve) & (~df.store_nbr.isin(drop) if drop else True)
    tr, va = df[m_tr], df[m_va]
    pred = np.zeros(len(va))
    va_key = list(zip(va.store_nbr.values, va.family.values))
    idx_by_key = {}
    for i, k in enumerate(va_key):
        idx_by_key.setdefault(k, []).append(i)
    for key, g in tr.groupby(["store_nbr", "family"], observed=True):
        if key not in idx_by_key:
            continue
        gv = va.iloc[idx_by_key[key]]
        if len(g) < 60 or g["y"].std() < 1e-6:      # 履歴が短い/定数の系列は平均で埋める
            pred[idx_by_key[key]] = g["y"].mean() if len(g) else 0.0
            continue
        w = 0.5 ** ((vs - g["date"]).dt.days.values / 365)   # 直近重み（e14 で採用済み）
        model = Ridge(alpha=3.0)
        model.fit(design(g), g["y"].values, sample_weight=w)
        pred[idx_by_key[key]] = model.predict(design(gv))
    pred = np.clip(pred, 0, None)
    yv = va["y"].values
    s = float(np.sqrt(np.mean((pred - yv) ** 2)))
    res[fl] = (s, pred, yv, va.index)
    print(f"{fl}  Ridge脚 単体 RMSLE {s:.4f}  {time.time()-t0:.0f}s", flush=True)
    np.save(c.OUTPUT / f"e17_ridge_{fl[0]}.npy", pred)
    np.save(c.OUTPUT / f"e17_y_{fl[0]}.npy", yv)
print("\n参考: LightGBM 単体は fold A 0.3841 / fold B 0.3815（新CV 平均 0.3828）")
print("次: e18 で LightGBM 予測とブレンドし、誤差相関と最適重みを測る")
