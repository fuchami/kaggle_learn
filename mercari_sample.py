# coding:utf-8

# %%  
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.model_selection import train_test_split
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import numpy as np

# %% データタイプを指定
types_dict_train = {'train_id': 'int64', 'item_condition_id':'int8', 'price':'float64', 'shipping':'int8'}
types_dict_test = {'test_id':'int64', 'item_condition_id':'int8', 'shipping':'int8'}

# %% tsvファイルからPandas DataFrameへ読み込み
train = pd.read_csv('./mercari/train.tsv', delimiter='\t', low_memory=True, dtype=types_dict_test)
test = pd.read_csv('./mercari/test.tsv', delimiter='\t', low_memory=True, dtype=types_dict_test)

# %% trainとtestのデータフレームの冒頭5行を表示させる
train.head()
test.head()

# trainとtestのサイズを確認
train.shape, test.shape

# %% データの統計量を確認
def display_all(df):
    with pd.option_context("display.max_rows", 1000):
        with pd.option_context("display.max_columns", 1000):
            display(df)

# trainの基本統計を表示
display_all(train.describe(include='all').transpose())

# %% trainのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する
train.category_name = train.category_name.astype('category')
train.item_description = train.item_description.astype('category')
train.name = train.name.astype('category')
train.brand_name = train.brand_name.astype('category')

#  testのカテゴリ名、商品説明、投稿タイトル、ブランド名のデータタイプを「category」へ変換する
test.category_name = test.category_name.astype('category')
test.item_description = test.item_description.astype('category')
test.name = test.name.astype('category')
test.brand_name = test.brand_name.astype('category')

# dtypesでデータ形式を確認
train.dtypes, test.dtypes