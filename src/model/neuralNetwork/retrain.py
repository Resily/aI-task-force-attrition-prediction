import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, log_loss, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import optuna
import json
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def retrain_model(df, best_params, seedValue=1000, output_file="retrain_evaluation_results.txt", model_file="retrain_model.pth"):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["label"]),
        df["label"],
        test_size=0.3,
        random_state=seedValue,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=seedValue)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    input_size = X_train.shape[1]
    model = NeuralNet(input_size, best_params['units1'], best_params['dropout1'], best_params['units2'], best_params['dropout2'], best_params['units3'], best_params['dropout3'], best_params['units4'], best_params['dropout4'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'])
    
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

    for epoch in range(best_params['epochs']):
        train_model(model, criterion, optimizer, train_loader, device)
    
    y_pred, y_true = evaluate_model(model, test_loader, device)
    y_pred_proba = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().detach().numpy()

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    logloss = log_loss(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)

    with open(output_file, "a") as f:
        f.write(f"Test Accuracy: {accuracy}\n")
        f.write(f"Test AUC: {auc}\n")
        f.write(f"Test F1 Score: {f1}\n")
        f.write(f"Test LogLoss: {logloss}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n\n")

    torch.save(model.state_dict(), model_file)  # モデルを保存

if __name__ == "__main__":
    df2 = pd.read_csv("../../data/processed/train_label_conversion_3.csv")
    seedValue = 1000

    datasets = {
        "3_other": df2,
    }

    for name, df in datasets.items():
        print(f"------------------\n{name}")
        best_params = tuning(X_train=df.drop(columns=["label"]), y_train=df["label"], num_splits=5, random_state=seedValue, n_trials=200)
        
        with open(f'best_params_nn_{name}.json', 'w') as f:
            json.dump(best_params, f)
        
        print("再学習")
        retrain_model(df, best_params, seedValue, output_file=f"retrain_evaluation_results_{name}.txt", model_file=f"retrain_model_{name}.pth")