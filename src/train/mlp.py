"""This file trains four models: MLP"""

#=============imports==============
from pathlib import Path
import matplotlib.pyplot as plt
import joblib
from src import data_prep
import pandas as pd

#=============MLP-imports==============
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

#=============metrics-import==============
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

base_dir = Path(__file__).resolve().parent.parent.parent
#=============MLP-Structure==============
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            # =============first layer_64-neuron==============
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            # =============second layer_32-neuron==============
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            # =============last layer_1-neuron==============
            nn.Linear(32, 1),
        )

    # =============forward-function==============
    def forward(self, x):
        return self.network(x)

#=============mlp-object==============
mlp = MLP()

#=============Loss==============
criterion = nn.BCEWithLogitsLoss()

#=============Adam-optimizer-used==============
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

#=============sum of MLP parameters==============
total_params = sum(p.numel() for p in mlp.parameters())
print(f"sum of MLP parameters: {total_params}")

# ============================================================
#               load data from data_prep.py
# ============================================================
Xtrain, Xval, ytrain, yval = data_prep.loader(
    show_print=False,
    return_test=False
)

#=============scaler==============
#TODO: try except
scaler_path = base_dir /"models" /"scaler.pkl"
scaler = joblib.load(scaler_path)

Xtrain_scaled = scaler.transform(Xtrain)
Xval_scaled = scaler.transform(Xval)

#=============turn-data to Tensor==============
X_train_tensor = torch.tensor(Xtrain_scaled, dtype=torch.float32)
x_validation_tensor = torch.tensor(Xval_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(ytrain, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(yval, dtype=torch.float32).reshape(-1, 1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#=============set batch size and loader==============
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

#=============loop control variables==============
epochs = 40
loss_history = []
loss_history_batch = []

#=============early stopping variables==============
patience = 5
counter = 0
best_val_loss = float("inf")

#=============Training-Loop==============
for epoch in range(epochs):
    total_loss = 0
    # =============return a Batch in each iter==============
    for X_batch, y_batch in train_loader:
        prediction = mlp(X_batch)
        # =============Calculate-Loss==============
        loss = criterion(prediction, y_batch)

        #=============zero-last-gradient==============
        optimizer.zero_grad()

        #=============Backpropagation==============
        loss.backward()

        # =============Update==============
        optimizer.step()

        total_loss += loss.item()
        # =============batch-loss==============
        loss_history_batch.append(loss.item())
    # =============average-loss==============
    avg_loss = total_loss / len(train_loader)

    # =============early-stopping==============
    mlp.eval()
    with torch.no_grad():
        val_out = mlp(x_validation_tensor)
        val_loss = criterion(val_out, y_test_tensor).item()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    mlp.train()
    loss_history.append(avg_loss)
    # =============report each 5 step loss ==============
    if (epoch + 1) % 5 == 0:

        print(f"Epoch {epoch+1}/{epochs}, " f"Loss: {avg_loss:.4f}")

# =============activate evaluating mode==============
mlp.eval()
results_path = base_dir /"reports"/"mlp"
results_path.mkdir(parents=True, exist_ok=True)

thresholds = [0.3, 0.5, 0.7]
all_results = []
# =============activate predicting mode==============
with torch.no_grad():
    # =============validation-proba==============
    outputs = mlp(x_validation_tensor)
    proba_mlp = torch.sigmoid(outputs).cpu().numpy().ravel()

    # =============train-proba==============
    outputs_train = mlp(X_train_tensor)
    proba_train_mlp = torch.sigmoid(outputs_train).cpu().numpy().ravel()

# =============search-thresholds==============
for threshold in thresholds:
    ypred_mlp = (proba_mlp >= threshold).astype(int)
    ypred_train_mlp = (proba_train_mlp >= threshold).astype(int)

    # ================================================
    #            confusion matrix train
    # ================================================
    cm_train = confusion_matrix(ytrain, ypred_train_mlp)
    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm_train,
        display_labels=["legitimate", "fraud"],
    )
    dsp.plot()
    plt.title(f"MLP train threshold: {threshold}")

    plt.savefig(
        results_path/ f"mlp_cm_train{int(threshold * 10)}.png"
    )
    plt.close()

    # ================================================
    #          confusion matrix validation
    # ================================================
    cm = confusion_matrix(yval, ypred_mlp)
    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["legitimate", "fraud"],
    )
    dsp.plot()
    plt.title(f"MLP threshold: {threshold}")

    plt.savefig(
        results_path/f"mlp_cm{int(threshold * 10)}.png"
    )
    plt.close()

    # ================================================
    #     Save train and validation scores to CSV
    # ================================================
    results = pd.DataFrame(
        [
            {
                # =============train==============
                "model": f"MLP tr = {threshold}",
                "dataset": "train",
                "precision": precision_score(ytrain, ypred_train_mlp),
                "recall": recall_score(ytrain, ypred_train_mlp),
                "f1_score": f1_score(ytrain, ypred_train_mlp),
                "accuracy": accuracy_score(ytrain, ypred_train_mlp),
            },
            {
                # =============validation==============
                "model": f"MLP tr = {threshold}",
                "dataset": "validation",
                "precision": precision_score(yval, ypred_mlp),
                "recall": recall_score(yval, ypred_mlp),
                "f1_score": f1_score(yval, ypred_mlp),
                "accuracy": accuracy_score(yval, ypred_mlp),
            },
        ]
    )
    # =============Save-Reports==============
    results.to_csv(
        results_path/f"mlp_scores{int(threshold * 10)}.csv",
        index=False,
    )

    all_results.append(
        {
            "threshold": threshold,
            "f1_score": f1_score(yval, ypred_mlp),
        }
    )

    print(f"\nMLP threshold {threshold} scores saved successfully.")
# =============best-threshold==============
f1_list = [res["f1_score"] for res in all_results]
best_idx = f1_list.index(max(f1_list))
best_threshold = all_results[best_idx]["threshold"]

print(f"\nBest MLP threshold based on validation F1: {best_threshold}")

# =============Save-Model==============
model_path = base_dir /"models"
model_path.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "state_dict": mlp.state_dict(),
        "threshold": best_threshold,
        "input_dim": 30,
    },
    model_path/"mlp.pt",
)
