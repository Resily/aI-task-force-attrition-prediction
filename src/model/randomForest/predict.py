"""
人事から共有されたインサイトデータを元に予測を行う。
"""
import pandas as pd
import joblib

def predict_new_data(X_new, model_path, ageNumber, seedValue=1000):    
    model = joblib.load(model_path)
    if '受検No.' in X_new.columns:
        exam_no = X_new['受検No.']
        X_new = X_new.drop(columns=['受検No.'])
    else:
        exam_no = pd.Series([None] * len(X_new))
    
    prediction_proba = model.predict_proba(X_new)[:, 1]
    
    prediction_proba_percentage = (prediction_proba * 100).round(2)
    
    risk_category = pd.cut(prediction_proba_percentage, 
                           bins=[0, 25, 50, 75, 100], 
                           labels=["低い", "やや低い", "やや高い", "高い"], 
                           right=False)

    feature_importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        '特徴量': X_new.columns,
        '重要度': feature_importances
    }).sort_values(by='重要度', ascending=False).head(200)

    # 各配列の長さを取得
    len_exam_no = len(exam_no)
    len_prediction_proba_percentage = len(prediction_proba_percentage)
    len_risk_category = len(risk_category)

    # 最大の長さを取得
    max_len = max(len_exam_no, len_prediction_proba_percentage, len_risk_category)

    # 各配列の長さを最大の長さに合わせる
    exam_no_list = exam_no.tolist()
    prediction_proba_percentage_list = prediction_proba_percentage.tolist()
    risk_category_list = risk_category.tolist()

    exam_no_list.extend([''] * (max_len - len_exam_no))
    prediction_proba_percentage_list.extend([''] * (max_len - len_prediction_proba_percentage))
    risk_category_list.extend([''] * (max_len - len_risk_category))

    results = pd.DataFrame({
        '受検No.': exam_no_list,
        f"{ageNumber}年以内に離職する確率(%)": prediction_proba_percentage_list,
        f"{ageNumber}年以内の離職リスク": risk_category_list
    })
    
    return results, feature_importance_df


if __name__ == "__main__":
    import datetime
    import os
    import re

    # 複数のCSVファイルパスをリストで指定
    csv_files = [
        "../../data/processed/predict/processed_predict_data_70023-20250303-163957295.csv",
        "../../data/processed/predict/processed_predict_data_70023-20250306-152752641.csv",
        "../../data/processed/predict/processed_predict_data_70023-20250310-165026091.csv",
    ]

    model = "../randomForest/model_downsampling_2_other.pkl"
    # model = "../randomForest/model_smoted_3_other.pkl"

    for new_data in csv_files:
        X_new = pd.read_csv(new_data)
        
        extracted_part = (re.search(r'(\d{5}-\d{8}-\d{9})', new_data)).group(1)
        print(f"抽出された部分: {extracted_part}")

        ageNumber = re.search(r'model_smoted_(\d)_other', model)
        # extracted_ageNumber = ageNumber.group(1)
        extracted_ageNumber = "2"

        results, feature_importance_df = predict_new_data(X_new, model, extracted_ageNumber)
        
        filename = f"result/【結果格納済み】{extracted_part}.xlsx"
        with pd.ExcelWriter(filename) as writer:
            results.to_excel(writer, sheet_name="予測結果", index=False)
            feature_importance_df.to_excel(writer, sheet_name="特徴量重要度", index=False)
        
        print(f"予測結果が出力されました: {filename}")