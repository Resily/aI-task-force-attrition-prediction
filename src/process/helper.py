import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import cross_val_score, KFold, train_test_split, StratifiedKFold

def period_to_year(term):
    if isinstance(term, str):
        year, half = term[:4], term[4:]
        return int(year) + (0.5 if half == '後期' else 0)
    return None

def label_conversion(row, thresholds, labels):
    if pd.isna(row['離職年']):
        for threshold, label in zip(thresholds, labels):
            if row['在職期間'] <= threshold:
                return label
        return labels[-1]
    else:
        for threshold, label in zip(thresholds, labels):
            if row['在職期間'] <= threshold:
                return label
        return labels[-1]

def label_conversion_3_6_9_other(row):
    return label_conversion(row, [3, 6, 9], ['離職3年以内', '離職3年より大きく6年以内', '離職6年より大きく9年以内', '離職or在職9年以降'])

def label_conversion_3(row):
    return label_conversion(row, [3], ['離職3年以内', '離職3年以降もしくは在職3年以降'])

def label_conversion_2(row):
    return label_conversion(row, [2], ['離職2年以内', '離職2年以降もしくは在職2年以降'])

def label_conversion_5(row):
    return label_conversion(row, [5], ['離職5年以内', '離職5年以降もしくは在職5年以降'])

def label_conversion_6(row):
    return label_conversion(row, [6], ['離職6年以内', '離職6年以降もしくは在職6年以上'])

def label_conversion_1_2_3_6_other(row):
    return label_conversion(row, [1, 2, 3, 6], ['離職1年以内', '離職1年より大きく2年以内', '離職2年より大きく3年以内', '離職または在職3年以降6年以内', '離職または在職6年以降'])

def label_conversion_1h_3_6_other(row):
    return label_conversion(row, [1.5, 3, 6], ['離職1.5年以内', '離職1.5年より大きく3年以内', '離職または在職3年以降6年以内', '離職または在職6年以降'])

def label_conversion_2_6_other(row):
    return label_conversion(row, [2, 6], ['離職2年以内', '離職2年以降6年以内', '離職または在職6年以降'])

def label_conversion_2_6_10_other(row):
    return label_conversion(row, [2, 6, 10], ['離職2年以内', '離職2年以降6年以内', '離職または在職6年以降', '離職または在職10年以降'])