"""e02: 単一 LightGBM（全1782系列を1モデル、日付基準 lag>=16）。
比較対象 = e01 dow4_mean (CV 0.4540)。
early stopping を valid で行うため best_iter 選択分の楽観バイアスがある（best_iter も記録）。
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c, features as F
from config import PARAMS, TRAIN_START as _TS

TRAIN_START = pd.Timestamp(_TS)

def main():
    df = F.build()
    df = df[df.date >= TRAIN_START - pd.Timedelta(days=1)]
    folds = c.make_folds(5)
    scores, iters, preds_all = [], [], []
    for k, (tr_end, vs, ve) in enumerate(folds):
        t0 = time.time()
        m_tr = (df.date >= TRAIN_START) & (df.date < tr_end) & df.sales.notna()
        m_va = (df.date >= vs) & (df.date <= ve)
        Xtr, ytr = df.loc[m_tr, F.FEATURES], df.loc[m_tr, "y"]
        Xva, yva = df.loc[m_va, F.FEATURES], df.loc[m_va, "y"]
        ds_tr = lgb.Dataset(Xtr, ytr, categorical_feature=F.CATS, free_raw_data=False)
        ds_va = lgb.Dataset(Xva, yva, categorical_feature=F.CATS, reference=ds_tr)
        mdl = lgb.train(PARAMS, ds_tr, num_boost_round=3000, valid_sets=[ds_va],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        p = np.clip(mdl.predict(Xva, num_iteration=mdl.best_iteration), 0, None)
        s = float(np.sqrt(np.mean((p - yva.values) ** 2)))
        scores.append(s); iters.append(mdl.best_iteration)
        v = df.loc[m_va, ["date", "store_nbr", "family", "y"]].copy(); v["pred"] = p; v["fold"] = k
        preds_all.append(v)
        print(f"fold{k} {vs.date()}..{ve.date()}  RMSLE {s:.4f}  best_iter {mdl.best_iteration}  n_tr {m_tr.sum():,}  {time.time()-t0:.0f}s", flush=True)

    print(f"\n== e02 LightGBM mean RMSLE {np.mean(scores):.4f} (fold: {[round(x,4) for x in scores]}) ==")
    print(f"   e01 dow4_mean          0.4540  → 差 {np.mean(scores)-0.4540:+.4f}")
    print(f"   best_iter: {iters}")

    oof = pd.concat(preds_all)
    oof.to_parquet(c.OUTPUT / "e02_oof.parquet", index=False)
    print("\n== family別 RMSLE 上位（誤差寄与順）==")
    oof["se"] = (oof.pred - oof.y) ** 2
    t = oof.groupby("family", observed=True).se.agg(["sum", "count"])
    t["rmsle"] = np.sqrt(t["sum"] / t["count"]); t["SSE比"] = t["sum"] / t["sum"].sum()
    print(t.sort_values("SSE比", ascending=False).head(8)[["rmsle", "SSE比"]].round(4).to_string())

    print("\n== 特徴重要度 top20 (gain) ==")
    imp = pd.Series(mdl.feature_importance("gain"), index=F.FEATURES).sort_values(ascending=False)
    print((imp / imp.sum()).head(20).round(4).to_string())

    # 全期間で再学習して test 予測（iter は fold の平均）
    n_est = int(np.mean(iters))
    m_all = (df.date >= TRAIN_START) & (~df.is_test) & df.sales.notna()
    mdl_f = lgb.train(PARAMS, lgb.Dataset(df.loc[m_all, F.FEATURES], df.loc[m_all, "y"],
                                          categorical_feature=F.CATS), num_boost_round=n_est)
    te_mask = df.is_test
    pt = np.expm1(np.clip(mdl_f.predict(df.loc[te_mask, F.FEATURES]), 0, None))
    sub = df.loc[te_mask, ["id"]].copy(); sub["sales"] = pt
    sub = sub.sort_values("id")
    te = pd.read_csv(c.DATA / "test.csv")
    c.make_submission(te, sub.set_index("id").loc[te.id, "sales"].values, "e02_lgbm", message="e02 single lightgbm")


if __name__ == "__main__":
    main()
