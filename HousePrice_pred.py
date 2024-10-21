#!/usr/bin/python
import pandas as pd
df = pd.read_csv('train.csv')
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from lightgbm import log_evaluation, early_stopping
import numpy as np
from sklearn.model_selection import GridSearchCV

Xy_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

Xy_all = pd.concat([Xy_train, X_test], axis=0)

cat_features = Xy_all.columns[Xy_all.dtypes == 'object']

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder(
    dtype = np.int32,
    handle_unknown = 'use_encoded_value',
    unknown_value = -1,
    encoded_missing_value=-1
).set_output(transform='pandas')

Xy_all[cat_features] = ordinal_encoder.fit_transform(Xy_all[cat_features])

X_test = Xy_all[Xy_all['SalePrice'].isna()].drop(columns=['SalePrice'])
Xy_train = Xy_all[-Xy_all['SalePrice'].isna()]

X_train = Xy_train.drop(columns='SalePrice')
y_train = Xy_train['SalePrice']

# # 默认LGBM
# M = lgb.LGBMRegressor(
#     boosting_type='gbdt',
#     num_leaves=31,
#     max_depth=-1,
#     learning_rate=0.1,
#     n_estimators=100,
#     subsample_for_bin=200000,
#     objective=None,
#     class_weight=None,
#     min_split_gain=0.0,
#     min_child_weight=0.001,
#     min_child_samples=20,
#     subsample=1.0,
#     subsample_freq=0,
#     colsample_bytree=1.0,
#     reg_alpha=0.0,
#     reg_lambda=0.0,
#     random_state=None,
#     n_jobs=None,
#     importance_type='split'
# )
#

# 优化后(没用)
M = lgb.LGBMRegressor(
    objective='regression',
    num_leaves=70,
    learning_rate=0.005,
    n_estimators=829,
    max_depth=7,
    metric='rmse',
    bagging_fraction=0.8,
    feature_fraction=0.8
)
M.fit(X_train, y_train)
y_pred = M.predict(X_test)


pd.DataFrame({
    'Id' : X_test['Id'],
    'SalePrice' : y_pred
}).to_csv('lgb_v1.csv', index=False)
