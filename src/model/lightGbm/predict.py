import pandas as pd
import json
import joblib
import lightgbm as lgb

def predict_new_data(new_data_path, model_path, ageNumber, seedValue=1000):
    new_data = pd.read_csv(new_data_path)

    X_new = new_data.drop(columns=["受検No."])
    
    model = joblib.load(model_path)
    
    prediction_proba = model.predict_proba(X_new)[:, 1]
    
    # 予測確率をパーセンテージに変換し、少数第2位まで表示
    prediction_proba_percentage = (prediction_proba * 100).round(2)
    
    results = pd.DataFrame({
        "受検No.": new_data["受検No."],
        f"{ageNumber}年以内に離職する確率(%)": prediction_proba_percentage
    })
    
    return results


if __name__ == "__main__":
    import datetime
    import os

    new_data = "../../data/processed/processed_predict_data_20250131_141715.csv"
    model = "../lightBgm/model_2_other.pkl"  # モデルパスを更新
    ageNumber = "2"
    
    model_name = os.path.basename(model).replace("lgb_", "").replace("_model.pkl", "")
    results = predict_new_data(new_data, model, ageNumber)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"result/予測結果_{model_name}_{timestamp}.xlsx"
    results.to_excel(filename, index=False)
    print("予測結果が出力されました")