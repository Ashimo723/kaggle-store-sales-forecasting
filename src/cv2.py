"""新 CV（seasonal 2fold）。test の季節位置を含み、既知の事故を除いた検証設計。

fold A: 2017-07-31..08-15  直近16日・全店（test 直前、学習量最大）
fold B: 2016-08-16..08-31  test と同じ日付窓。**store 18/25 を valid から除外**
        （e13: この2店舗の一時休業が素の fold の SSE の 68.8% を占め、除くと 0.6763→約0.378）

判定は「2fold 平均」だけで決めず、**2 fold の符号一致**も条件にする（2本は分散が大きい）。
"""
import pandas as pd

FOLDS = [
    ("A:2017-08前半", pd.Timestamp("2017-07-31"), pd.Timestamp("2017-08-15"), set()),
    ("B:2016-08後半", pd.Timestamp("2016-08-16"), pd.Timestamp("2016-08-31"), {18, 25}),
]
