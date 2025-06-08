import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, KFold, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns
import optuna

import helper
import params

# Windowsの場合（MS ゴシックを使用）
# plt.rcParams['font.family'] = 'MS Gothic'
# フォントを変更(macの時のみ）
plt.rcParams['font.family'] = 'Hiragino Sans'

# Macの場合（ヒラギノ角ゴシックを使用）
# plt.rcParams['font.family'] = 'Hiragino Maru Gothic Pro'

    
def preprocessing(df, conversion_method):
    def apply_common_transformations(df):
        df['入社年'] = df['入社時期'].apply(helper.period_to_year)
        df['離職年'] = df['離職時期'].apply(helper.period_to_year)
        df['在職期間'] = df.apply(lambda row: params.current_year - row['入社年'] if pd.isna(row['離職年']) else row['離職年'] - row['入社年'], axis=1)
        return df

    conversion_methods = {
        'label_conversion_3': helper.label_conversion_3,
        'label_conversion_2': helper.label_conversion_2,
        'label_conversion_5': helper.label_conversion_5,
        'label_conversion_3_6_9_other': helper.label_conversion_3_6_9_other,
        'label_conversion_6': helper.label_conversion_6,
        'label_conversion_1_2_3_6_other': helper.label_conversion_1_2_3_6_other,
        'label_conversion_1h_3_6_other': helper.label_conversion_1h_3_6_other,
        'label_conversion_2_6_10_other': helper.label_conversion_2_6_10_other,
        'label_conversion_2_6_other': helper.label_conversion_2_6_other
    }

    if conversion_method in conversion_methods:
        df = apply_common_transformations(df)
        df['label'] = df.apply(conversion_methods[conversion_method], axis=1)
        df = df[df['label'] != '削除対象']
    else:
        print('conversion_methodが違います。')
        return

    df = process_unnecessary_columns(df)

    # 正解ラベルの作成
    df = df.drop(columns=['離職時期', '入社時期', '入社年', '離職年', '在職期間'])
    label_replacements = {
        'label_conversion_3': {'離職3年以内': 1, '離職3年以降もしくは在職3年以降': 0},
        'label_conversion_2': {'離職2年以内': 1, '離職2年以降もしくは在職2年以降': 0},
        'label_conversion_5': {'離職5年以内': 1, '離職5年以降もしくは在職5年以降': 0},
        'label_conversion_6': {'離職6年以内': 1, '離職6年以降もしくは在職6年以上': 0},
        'label_conversion_3_6_9_other': {'離職3年以内': 3, '離職3年より大きく6年以内': 2, '離職6年より大きく9年以内': 1, '離職or在職9年以降': 0},
        'label_conversion_1_2_3_6_other': {'離職1年以内': 4, '離職1年より大きく2年以内': 3, '離職2年より大きく3年以内': 2, '離職または在職3年以降6年以内': 1, '離職または在職6年以降': 0},
        'label_conversion_1h_3_6_other': {'離職1.5年以内': 3, '離職1.5年より大きく3年以内': 2, '離職または在職3年以降6年以内': 1, '離職または在職6年以降': 0},
        'label_conversion_2_6_10_other': {'離職2年以内': 3, '離職2年以降6年以内': 2, '離職または在職6年以降10年以内': 1, '離職または在職10年以降': 0},
        'label_conversion_2_6_other': {'離職2年以内': 2, '離職2年以降6年以内': 1, '離職または在職6年以降': 0}
    }

    if conversion_method in label_replacements:
        df['label'] = df['label'].replace(label_replacements[conversion_method])

    return df

def process_unnecessary_columns(df):
    df['性別'].fillna('不明', inplace=True)
    df['性別'] = df['性別'].replace({'男性': 0, '女性': 1, '不明': 2})
    drop_columns = df.columns[df.columns.str.contains('役職_|パフォーマンス評価_|_段階|_順位|_矢印|_未回答|総合評価_|応答態度_|検査セクション')]
    return df.drop(columns=drop_columns)