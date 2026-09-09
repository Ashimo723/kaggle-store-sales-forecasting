"""cv3: rolling origin CV（27 fold）。CV 再設計。

なぜ作り直すか:
  旧cv2 は 2fold。e23 で測った fold 間変動は sd 0.02 で、2本の標準誤差は 0.014。
  判定したい効果 0.002 の7倍あり、**偶然で符号が決まる**。実際 hl=180（旧5fold採用→新2foldで逆転）と
  hl=365（新2fold採用→LB で +0.0215 悪化）で2度誤った。

設計:
  - 2016-06-01 から 16日ステップで 2017-08-15 まで、重複なしの16日ブロックを valid にする
  - 各 fold: train = date < valid_start（提出時の「学習終端 = test 直前」と同じ構造）
  - 27 fold の標準誤差は 0.02/√27 ≈ 0.004。さらに**対比較**（同一 fold での差分）なら fold 変動が相殺され
    はるかに小さくなるので、判定は必ず「同一 fold での差」で行う
  - 判定基準: **|t| >= 3**（提出#2/#3 の実測で t=-3.68 は転移し t=-1.76 は逆転した）。
    平均差の大きさでは決めない。勝率と中央値も併記し、少数の異常 fold が平均を作っていないか確認する
  - 注意: 12/25 は train に行が存在しない（欠測日）。valid マスクは必ず sales.notna() で絞ること
"""
import pandas as pd

def folds(start="2016-06-01", end="2017-08-15", horizon=16, step=16):
    out, s = [], pd.Timestamp(start)
    end = pd.Timestamp(end)
    while s + pd.Timedelta(days=horizon - 1) <= end:
        out.append((f"{s.date()}", s, s + pd.Timedelta(days=horizon - 1)))
        s = s + pd.Timedelta(days=step)
    return out

FOLDS = folds()
if __name__ == "__main__":
    print(f"{len(FOLDS)} folds: {FOLDS[0][0]} .. {FOLDS[-1][0]}")
