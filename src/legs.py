"""異質な脚（予測器）の定義。LightGBM とは情報源・帰納バイアスが違うものを揃える。"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
import features as F

K_FOURIER = 3

def _design(d):
    X = [np.ones(len(d))]
    dow = d["dow"].values
    for k in range(1, 7):
        X.append((dow == k).astype(float))
    doy = d["dayofyear"].values / 365.25
    for k in range(1, K_FOURIER + 1):
        X.append(np.sin(2 * np.pi * k * doy)); X.append(np.cos(2 * np.pi * k * doy))
    X.append(d["is_payday"].values.astype(float))
    X.append(d["days_since_payday"].values / 15.0)
    X.append(d["hol_any"].values.astype(float))
    X.append(d["is_nye"].values.astype(float))
    X.append(np.log1p(d["onpromotion"].values))
    X.append((d["date"].values.astype("datetime64[D]").astype(int) - 16000) / 1000.0)
    return np.column_stack(X)


def ridge_leg(tr, va, alpha=3.0, min_hist=60):
    """系列ごとの Ridge（曜日/年次フーリエ/給与日/祝日/プロモ/線形トレンド）。ラグを一切使わない。"""
    pred = np.zeros(len(va))
    idx = {}
    for i, k in enumerate(zip(va.store_nbr.values, va.family.values)):
        idx.setdefault(k, []).append(i)
    for key, g in tr.groupby(["store_nbr", "family"], observed=True):
        if key not in idx: continue
        rows = idx[key]
        if len(g) < min_hist or g["y"].std() < 1e-6:
            pred[rows] = g["y"].mean() if len(g) else 0.0
            continue
        m = Ridge(alpha=alpha).fit(_design(g), g["y"].values)
        pred[rows] = m.predict(_design(va.iloc[rows]))
    return np.clip(pred, 0, None)


def dow_mean_leg(va):
    """同曜日4週平均（e01 の best ベースライン、ホライズン整合版）。特徴として既に持っているが単体脚として測る。"""
    return np.clip(va["dow4_mean"].values, 0, None)


def seasonal_naive_leg(va):
    """前年同期（lag364）。年次季節性だけを見る脚。"""
    return np.clip(np.nan_to_num(va["lag364"].values, nan=0.0), 0, None)


def croston_leg(tr, va, alpha=0.1):
    """間欠需要向け Croston 法。非ゼロ需要の大きさと発生間隔を別々に指数平滑する。
    e24 で見つけたゼロ率「中」帯(RMSLE 0.63)に効く可能性があり、GBDT と帰納バイアスが最も違う。"""
    pred = np.zeros(len(va))
    idx = {}
    for i, k in enumerate(zip(va.store_nbr.values, va.family.values)):
        idx.setdefault(k, []).append(i)
    for key, g in tr.groupby(["store_nbr", "family"], observed=True):
        if key not in idx: continue
        y = g["y"].values
        nz = np.where(y > 1e-9)[0]
        if len(nz) < 3:
            pred[idx[key]] = 0.0
            continue
        z = y[nz[0]]          # 需要の大きさ
        p = 1.0               # 発生間隔
        last = nz[0]
        for i in nz[1:]:
            z = alpha * y[i] + (1 - alpha) * z
            p = alpha * (i - last) + (1 - alpha) * p
            last = i
        pred[idx[key]] = z / max(p, 1e-9)
    return np.clip(pred, 0, None)
