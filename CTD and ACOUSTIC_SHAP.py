#機械学習(XGBoost)を用いた親潮黒潮混合域の生態系に影響を与えやすい環境要因の分析
#卒業研究で使用したPythonのコードになります。SHAPと呼ばれる、特徴量の目的変数への影響度を可視化できるツールを用いて分析を行なっています。

import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import shap


#生態系の生物量データと環境データをまとめたファイルをインポート(データは大学の都合でお見せできないためXXXXとしています。)
df = pd.read_csv("XXXXXXXXXX.csv")
#ファイルの中から必要な特徴量を抽出(NASCが生物量データ、それ以外が環境データ)
df = df.loc[:, ["longitude","latitude","depth","temperature","turbidity","fluorescene","SPAR","salinity","density","NASC"]]

#K-分割交差検証
from sklearn.model_selection import KFold
kf = KFold(n_splits = 5, shuffle = True)
df["fold"] = -1
for fold, (train_idx, valid_idx) in enumerate(kf.split(df)):
    df.loc[valid_idx, "fold"] = fold

fold = 0
train = df.loc[df["fold"] != fold].copy()
valid = df.loc[df["fold"] == fold].copy()

feat_cols = train.drop(columns = ["fold", "NASC"]).columns.tolist()


#訓練データと評価データの定義 X(特徴量)が環境要因、Y(目的変数)が生物量
X_train = train[feat_cols]
X_valid = valid[feat_cols]
y_train = train["NASC"]
y_valid = valid["NASC"]

#XGboost用にデータを整形
dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_valid, y_valid)

#XGBoostの評価指標
params = {
    "objective" : "reg:squarederror",
    "eval_metric" : "rmse"
}

#XGboostの実行
model = xgb.train(
    params = params,
    dtrain = dtrain,
    evals = [(dtrain, "train"), (dvalid, "valid")],
    num_boost_round = 100,
    early_stopping_rounds = 10,
)


predt = model.predict(xgb.DMatrix(X_train))
pred = model.predict(xgb.DMatrix(X_valid))


#RMSE(二乗平均平方根誤差)の算出
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_valid, pred)
rmse = np.sqrt(mse)

#SHAP(特徴量の目的変数への寄与度を分析するツール)
explainer = shap.TreeExplainer(model=model, feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(X=X_train)

# SHAPの分布図
shap.summary_plot(shap_values, X_train)

# SHAPの散布図
shap.dependence_plot("fluorescene",shap_values,X_train, interaction_index="SPAR")

