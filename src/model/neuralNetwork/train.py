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

class NeuralNet(nn.Module):
    def __init__(self, input_size, units1, dropout1, units2, dropout2, units3, dropout3, units4, dropout4):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, units1)
        self.dropout1 = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(units1, units2)
        self.dropout2 = nn.Dropout(dropout2)
        self.fc3 = nn.Linear(units2, units3)
        self.dropout3 = nn.Dropout(dropout3)
        self.fc4 = nn.Linear(units3, units4)
        self.dropout4 = nn.Dropout(dropout4)
        self.fc5 = nn.Linear(units4, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        x = torch.sigmoid(self.fc5(x))
        return x

def create_model(trial, input_size):
    units1 = trial.suggest_int('units1', 32, 512)
    dropout1 = trial.suggest_float('dropout1', 0.2, 0.5)
    units2 = trial.suggest_int('units2', 32, 512)
    dropout2 = trial.suggest_float('dropout2', 0.2, 0.5)
    units3 = trial.suggest_int('units3', 32, 512)
    dropout3 = trial.suggest_float('dropout3', 0.2, 0.5)
    units4 = trial.suggest_int('units4', 32, 512)
    dropout4 = trial.suggest_float('dropout4', 0.2, 0.5)
    model = NeuralNet(input_size, units1, dropout1, units2, dropout2, units3, dropout3, units4, dropout4)
    return model

def train_model(model, criterion, optimizer, dataloader, device):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)

def tuning(X_train, y_train, num_splits=5, random_state=42, n_trials=200):
    def optuna_tuning(trial):
        input_size = X_train.shape[1]
        model = create_model(trial, input_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=trial.suggest_float('lr', 1e-6, 1e-1, log=True))
        
        skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
        cv_scores = []

        for train_index, valid_index in skf.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_tr, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]

            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
            
            train_dataset = TensorDataset(torch.tensor(X_tr.values, dtype=torch.float32), torch.tensor(y_tr.values, dtype=torch.float32))
            valid_dataset = TensorDataset(torch.tensor(X_val.values, dtype=torch.float32), torch.tensor(y_val.values, dtype=torch.float32))
            train_loader = DataLoader(train_dataset, batch_size=trial.suggest_int('batch_size', 16, 256), shuffle=True)
            valid_loader = DataLoader(valid_dataset, batch_size=trial.suggest_int('batch_size', 16, 256), shuffle=False)

            for epoch in range(trial.suggest_int('epochs', 100, 300)):
                train_model(model, criterion, optimizer, train_loader, device)
            
            y_pred, y_true = evaluate_model(model, valid_loader, device)
            cv_scores.append(f1_score(y_true, y_pred, average='binary'))
            print(f"Current CV Score: {sum(cv_scores) / len(cv_scores)}")
        return sum(cv_scores) / len(cv_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(optuna_tuning, n_trials=n_trials)

    best_params = study.best_trial.params

    return best_params

def evaluate_best_model(df, best_params, seedValue=1000, output_file="evaluation_results.txt", model_file="model.pth"):
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

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

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
    # df1 = pd.read_csv("../../data/processed/train_label_conversion_2.csv")
    df2 = pd.read_csv("../../data/processed/train_label_conversion_3.csv")
    # df3 = pd.read_csv("../../data/processed/train_downsample_label_conversion_2.csv")
    # df4 = pd.read_csv("../../data/processed/train_downsample_label_conversion_3.csv")

    seedValue = 1000

    datasets = {
        # "2_other": df1,
        "3_other": df2,
        # "downsampling_2_other": df3,
        # "downsampling_3_other": df4
    }

    for name, df in datasets.items():
        print(f"------------------\n{name}")
        best_params = tuning(X_train=df.drop(columns=["label"]), y_train=df["label"], num_splits=5, random_state=seedValue, n_trials=110)
        
        with open(f'best_params_nn_{name}.json', 'w') as f:
            json.dump(best_params, f)
        
        print("評価")
        evaluate_best_model(df, best_params, seedValue, output_file=f"evaluation_results_{name}.txt", model_file=f"model_{name}.pth")