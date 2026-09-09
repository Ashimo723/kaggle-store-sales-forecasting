"""モデル設定。スクリプト間で共有する定数はここに置く（import で副作用が起きないように）。"""
PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05, num_leaves=128,
              min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
              lambda_l2=1.0, num_threads=8, verbose=-1, seed=42)
TRAIN_START = "2015-01-01"
