# coding:utf-8
"""
タイタニックをやってみる
https://www.codexa.net/kaggle-titanic-beginner/
"""

# %%
import pandas as pd
import numpy as np

train = pd.read_csv('./titanic/train.csv')
test = pd.read_csv('./titanic/test.csv')

# 中身を確認
train.head()
test.head()

# %%
# 簡単な統計情報とサイズを確認
print(train.shape)
print(test.shape) 

# %%
# 各データセットの基本統計量
train.describe()
test.describe()

# %%
# データセットの欠損の確認
def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum()/len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(
        columns = {0: '欠損数', 1: '%'})
    return kesson_table_ren_columns

kesson_table(train)
kesson_table(test)

# %%
# データセットの事前前処理
# 欠損値には全データの中央値を代理データとする

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

kesson_table(train)

# %%
# 文字列データを数字に変換
train["Sex"] = train["Sex"].map({"male": 0, "female": 1})
train["Embarked"] = train["Embarked"].map({"S":0, "C":1, "Q":2})

train.head(10)

# %%
# テストデータも同様に変換
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()

test.head(10)

# %%
# 決定木を使って予測モデルを作成する
from sklearn import tree

# [train]の目的変数と説明変数の値を取得
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# 決定木の作成
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# [test]の説明変数の値を取得
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# [test]の説明変数を使って[my_tree_one]のモデルで予測
my_prediction = my_tree_one.predict(test_features)

# 予測データのサイズを確認
my_prediction.shape
# 予測データの中身を確認
print(my_prediction)

# %% 
# kaggleへの投稿のために書き出し
PassengerID = np.array(test["PassengerId"]).astype(int)

# 予測データとPassengerIDをデータフレームへ落とし込む
my_solution = pd.DataFrame(my_prediction, PassengerID, columns=["Survived"])

# my_tree_one.csvとして書き出し
# my_solution.to_csv("my_tree_one.csv", index_label=["PassengerId"])

# %%
# 説明変数を追加してやってみる
features_two = train[["Pclass", "Age","Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# 決定木の作成とアーギュメントの設定
max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=1)
my_tree_two = my_tree_two.fit(features_two, target)

# testから「その2」で使う項目の値を取り出す
test_feature_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# 「その2」の決定木を使って予測をしてCSVへ書き出し
my_prediction_tree_two = my_tree_two.predict(test_feature_2)
PassengerID = np.array(test["PassengerId"]).astype(int)

# 予測データとPassengerIDをデータフレームへ落とし込む
my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerID, columns=["Survived"])
my_solution_tree_two.to_csv("my_tree_two.csv", index_label=["PassengerId"])
