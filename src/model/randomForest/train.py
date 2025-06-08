import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import optuna
import json
import joblib

def tuning(X_train, y_train, num_splits=5, random_state=42, n_trials=150):
    def optuna_tuning(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 2, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 70),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 100),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }

        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
        cv_scores = []

        for train_index, valid_index in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]

            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

            model = RandomForestClassifier(**param, random_state=random_state, class_weight=class_weights_dict)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            cv_scores.append(f1_score(y_val, y_pred, average='binary'))
            print(f"Current CV Score: {sum(cv_scores) / len(cv_scores)}")
        return sum(cv_scores) / len(cv_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_tuning, n_trials=n_trials)

    best_params = study.best_trial.params
    return best_params, study.trials_dataframe()

def evaluate_model(df, best_params, seedValue=1000, output_file="evaluation_results.txt", model_file="model.pkl"):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["label"]),
        df["label"],
        test_size=0.3,
        random_state=seedValue,
    )

    model = RandomForestClassifier(**best_params, random_state=seedValue)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    with open(output_file, "a") as f:
        f.write(f"学習データのシェイプ: {X_train.shape}\n")
        f.write(f"Test Accuracy: {accuracy}\n")
        f.write(f"Test AUC: {auc}\n")
        f.write(f"Test F1 Score: {f1}\n")
        f.write(f"Test LogLoss: {logloss}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")
    joblib.dump(model, model_file)
    return accuracy, auc, f1, logloss

def confirm_overfitting(X_train, y_train, df_test, best_params, output_file="evaluation_results.txt", model_file="model.pkl"):
    X_test = df_test.drop(columns=["label"])
    y_test = df_test["label"]

    model = RandomForestClassifier(**best_params, random_state=seedValue)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    with open(output_file, "a") as f:
        f.write(f"[過学習確認]Test Accuracy: {accuracy}\n")
        f.write(f"[過学習確認]Test AUC: {auc}\n")
        f.write(f"[過学習確認]Test F1 Score: {f1}\n")
        f.write(f"[過学習確認]Test LogLoss: {logloss}\n")
        f.write("[過学習確認]Confusion Matrix:\n")
        f.write(f"{cm}\n\n")

    joblib.dump(model, model_file)
    return accuracy, auc, f1, logloss

def process_datasets(datasets, test_df, seedValue, isCountermeasuresOverfitting):
    for name, df in datasets.items():
        print(f"------------------\n{name}")
        best_params, cv_results = tuning(X_train=df.drop(columns=["label"]), y_train=df["label"], num_splits=5, random_state=seedValue, n_trials=100)
        
        with open(f'best_params_rf_{name}.json', 'w') as f:
            json.dump(best_params, f)
        
        print("評価")
        evaluate_model(df, best_params, seedValue, output_file=f"evaluation_results_{name}.txt", model_file=f"model_{name}.pkl")

        if isCountermeasuresOverfitting:
            print(f"過学習確認{name}")
            with open(f'best_params_rf_{name}.json', 'r') as f:
                best_params = json.load(f)
            confirm_overfitting(df.drop(columns=["label"]), df["label"], test_df, best_params, output_file=f"evaluation_results_{name}.txt", model_file=f"model_{name}.pkl")

if __name__ == "__main__":
    seedValue = 1000
    # isCountermeasuresOverfitting = True
    isCountermeasuresOverfitting = False
    # datasets_2 = {
    #     "2_other": pd.read_csv("../../data/processed/学習データ_label_conversion_2_過学習True.csv"),
    #     "downsampling_2_other": pd.read_csv("../../data/processed/train_downsample_label_conversion_2.csv"),
    #     "smoted_2_other": pd.read_csv("../../data/processed/学習データ_smoteオーバサンプリング_label_conversion_2_過学習True.csv"),
    # }
    # test_df_2 = pd.read_csv("../../data/processed/過学習確認用テストデータ_label_conversion_2.csv")
    # process_datasets(datasets_2, test_df_2, seedValue)

    # datasets_3 = {
    #     "3_other": pd.read_csv("../../data/processed/学習データ_label_conversion_3_過学習True.csv"),
    #     "downsampling_3_other": pd.read_csv("../../data/processed/学習データ_ダウンサンプリング_label_conversion_3_過学習True.csv"),
    #     "smoted_3_other": pd.read_csv("../../data/processed/学習データ_label_conversion_3_過学習True.csv")
    # }
    datasets_3 = {
        "3_other": pd.read_csv("../../data/processed/学習データ_label_conversion_3_過学習False.csv"),
        "downsampling_3_other": pd.read_csv("../../data/processed/学習データ_ダウンサンプリング_label_conversion_3_過学習False.csv"),
        "smoted_3_other": pd.read_csv("../../data/processed/学習データ_smoteオーバサンプリング_label_conversion_3_過学習False.csv")
    }
    test_df_3 = pd.read_csv("../../data/processed/過学習確認用テストデータ_label_conversion_3.csv")
    process_datasets(datasets_3, test_df_3, seedValue, isCountermeasuresOverfitting)