import pandas as pd
import joblib
import re
import os
import warnings
from pathlib import Path


# 警告抑制設定
# sklearn バージョン警告を抑制
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

# 定数定義
GENDER_MAPPING = {"男性": 0, "女性": 1, "不明": 2}
RISK_CATEGORIES = ["低い", "やや低い", "やや高い", "高い"]
RISK_BINS = [0, 25, 50, 75, 100]


def raw_data_process(new_data, extracted_part):
    """
    予測データの前処理を行う関数
    
    Args:
        new_data: 予測用の生データ
        extracted_part: データIDなどの識別子
        
    Returns:
        前処理済みのデータフレーム
    """
    print(f"{extracted_part}を加工します...")
    df_predict_data = new_data.copy()
    
    # カラム名の整形（複数行ヘッダーの結合、"Unnamed"を含む部分を除外）
    df_predict_data.columns = [
        "_".join(filter(lambda x: x and "Unnamed" not in str(x), map(str, col)))
        .replace("\n", "")
        .strip()
        for col in df_predict_data.columns.values
    ]
    
    # 不要なカラムの削除
    unwanted_columns = [
        "受検者種別", "企業コード", "取込み処理ID", "取込み順",
        "問題冊子番号", "氏名_姓", "氏名_名", "生年月日",
    ]
    df_predict_data = df_predict_data.drop(columns=unwanted_columns, errors="ignore")
    
    # 特定のパターンに一致する列を削除
    patterns = [
        "役職_", "パフォーマンス評価_", "_段階", "_順位", 
        "_矢印", "_未回答", "総合評価_", "応答態度_", "検査セクション",
    ]
    df_predict_data = df_predict_data.drop(
        columns=df_predict_data.filter(regex="|".join(patterns)).columns, 
        errors="ignore"
    )

    # 性別の処理 - 警告を避けるため inplace=True を使わない方法に変更
    if "性別" in df_predict_data.columns:
        # inplace=True の代わりに代入を使用
        df_predict_data["性別"] = df_predict_data["性別"].fillna("不明")
        # replace の警告を避けるためにマッピング方法を変更
        df_predict_data["性別"] = df_predict_data["性別"].map(GENDER_MAPPING).astype('int')

    # その他のカラム処理
    df_predict_data = df_predict_data.drop(columns=["出力", "No."], errors="ignore")
    
    # 年代を年齢に変更
    if "年代" in df_predict_data.columns:
        df_predict_data = df_predict_data.rename(columns={"年代": "年齢"})

    # 学習データとの比較（デバッグ用）
    try:
        base_dir = Path(__file__).parent.parent.parent  # src/modelの親ディレクトリ
        train_data_path = base_dir / "data" / "processed" / "train" / "学習データ_label_conversion_3_過学習False.csv"
        if os.path.exists(train_data_path):
            df = pd.read_csv(train_data_path)
            diff = set(df.columns.to_list()) - set(df_predict_data.columns.to_list())
            label_counts = df["label"].value_counts(normalize=True) if "label" in df.columns else None
            print(f"カラム数: {len(df_predict_data.columns)}")
            print(f"差分カラム: {diff}")
            if label_counts is not None:
                print(f"ラベル分布: {label_counts}")
        else:
            print(f"学習データファイルが見つかりません: {train_data_path}")
    except Exception as e:
        print(f"学習データ比較中にエラーが発生しました: {e}")

    print(f"{extracted_part}の加工完了")
    return df_predict_data


def predict_new_data(X_new, model_path, age_number, seed_value=1000):
    """
    新しいデータに対して予測を行う関数
    
    Args:
        X_new: 予測対象の前処理済みデータ
        model_path: モデルのパス
        age_number: 年数表示用
        seed_value: ランダムシード
        
    Returns:
        予測結果と特徴量重要度のDataFrame
    """
    # scikit-learn の警告を一時的に抑制
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        try:
            model = joblib.load(model_path)
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            raise

    # 受検No.の取り出し
    if "受検No." in X_new.columns:
        exam_no = X_new["受検No."].copy()
        X_new = X_new.drop(columns=["受検No."])
    else:
        exam_no = pd.Series([None] * len(X_new))

    # 予測実行
    try:
        # scikit-learn の警告を一時的に抑制
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            prediction_proba = model.predict_proba(X_new)[:, 1]
            prediction_proba_percentage = (prediction_proba * 100).round(2)
    except Exception as e:
        print(f"予測処理エラー: {e}")
        raise

    # リスクカテゴリの作成
    risk_category = pd.cut(
        prediction_proba_percentage,
        bins=RISK_BINS,
        labels=RISK_CATEGORIES,
        right=False,
    )

    # 特徴量重要度の取得
    feature_importances = model.feature_importances_
    feature_importance_df = (
        pd.DataFrame({"特徴量": X_new.columns, "重要度": feature_importances})
        .sort_values(by="重要度", ascending=False)
        .head(200)
    )

    # 結果データフレームの作成
    results = pd.DataFrame({
        "受検No.": exam_no.values,
        f"{age_number}年以内に離職する確率(%)": prediction_proba_percentage,
        f"{age_number}年以内の離職リスク": risk_category,
    })

    return results, feature_importance_df


def extract_model_age(model_path):
    """モデルパスから年数を抽出する関数"""
    match = re.search(r"model_smoted_(\d)_other", model_path)
    return match.group(1) if match else "3"


def ensure_directory(directory):
    """ディレクトリが存在しない場合に作成する関数"""
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == "__main__":
    # pandas の警告を抑制
    pd.options.mode.chained_assignment = None  # default='warn'
    
    # 基準ディレクトリの設定
    base_dir = Path(__file__).parent.parent.parent  # src/modelの親ディレクトリ
    
    # 複数のexcelファイルパスをリストで指定
    excel_files = [
        base_dir / "data" / "raw" / "predict" / "個人情報削除_0310-0316処理分_結果一覧表_株式会社アドバンテッジリスクマネジメント_70023-20250318-164706485.xlsx",
        base_dir / "data" / "raw" / "predict" / "個人情報削除_0317-0323処理分_結果一覧表_株式会社アドバンテッジリスクマネジメント_70023-20250324-143238167.xlsx", 
        base_dir / "data" / "raw" / "predict" / "個人情報削除_0324-0330処理分_結果一覧表_株式会社アドバンテッジリスクマネジメント_70023-20250401-150149211.xlsx",
    ]

    # モデルパスの設定
    model_path = Path(__file__).parent / "param" / "model_smoted_3_other.pkl"
    
    # 結果格納ディレクトリの確保
    result_dir = Path(__file__).parent / "result"
    ensure_directory(result_dir)

    for excel_path in excel_files:
        try:
            print("--------------------------------")
            print("処理を開始します。")
            X_new = pd.read_excel(excel_path, header=[0, 1, 2, 3, 4])
            
            # ファイル名からIDを抽出
            file_name = os.path.basename(excel_path)
            match = re.search(r"(\d{5}-\d{8}-\d{9})", file_name)
            
            if not match:
                print(f"ファイル名からIDを抽出できませんでした: {file_name}")
                continue
                
            extracted_part = match.group(1)
            print(f"抽出されたID: {extracted_part}")

            # 年数の抽出
            age_number = extract_model_age(str(model_path))
            
            # データ前処理
            processed_X_new = raw_data_process(X_new, extracted_part)
            
            # 予測実行
            results, feature_importance_df = predict_new_data(
                processed_X_new, model_path, age_number
            )
            
            # 結果保存
            result_path = result_dir / f"【結果格納済み】{extracted_part}.xlsx"
            with pd.ExcelWriter(result_path) as writer:
                results.to_excel(writer, sheet_name="予測結果", index=False)
                feature_importance_df.to_excel(writer, sheet_name="特徴量重要度", index=False)
            
            # print(f"予測結果の出力完了: {result_path}")
            
            # 処理済みデータの保存（デバッグ用）
            processed_data_dir = base_dir / "data" / "processed" / "predict"
            ensure_directory(processed_data_dir)
            processed_data_path = processed_data_dir / f"processed_predict_data_{extracted_part}.csv"
            processed_X_new.to_csv(processed_data_path, index=False)
            # print(f"処理済みデータ保存: {processed_data_path}")
            print("処理が完了しました。")
            print("--------------------------------")
            
        except Exception as e:
            print(f"処理中にエラーが発生しました: {e}")
