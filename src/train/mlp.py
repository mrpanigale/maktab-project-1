"""This file trains four models: MLP"""

# imports
import os
import matplotlib.pyplot as plt
import joblib
from src import data_prep
import pandas as pd

# MLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# metrics
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)


# MLP structure
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


mlp = MLP()
#  Loss function
criterion = nn.BCEWithLogitsLoss()
#  Optimizer
optimizer = optim.Adam(mlp.parameters(), lr=0.001)

total_params = sum(p.numel() for p in mlp.parameters())
print(f"sum of MLP parameters: {total_params}")
# ============================================================
# load data from data_prep.py
# ============================================================
Xtrain, Xval, Xtest_final, ytrain, yval, ytest_final = data_prep.loader(
    show_print=False
)

# load scaler object
scaler = joblib.load(r"E:\MLprojects\maktap-project-1\models\scaler.pkl")

Xtrain_scaled = scaler.transform(Xtrain)
Xval_scaled = scaler.transform(Xval)

# tensor X, y
X_train_tensor = torch.tensor(Xtrain_scaled, dtype=torch.float32)
x_validation_tensor = torch.tensor(Xval_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(ytrain, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(yval, dtype=torch.float32).reshape(-1, 1)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# return batchs
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

# training loop
epochs = 40
loss_history = []
loss_history_batch = []
#early stopping setup
patience = 5
counter = 0
best_val_loss = float('inf')


for epoch in range(epochs):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        prediction = mlp(X_batch)
        loss = criterion(prediction, y_batch)
        # zero last round gradient
        optimizer.zero_grad()
        # calculate gradient
        loss.backward()
        # update parameters
        optimizer.step()

        total_loss += loss.item()
        # each update loss
        loss_history_batch.append(loss.item())
    # average loss
    avg_loss = total_loss / len(train_loader)
    #extract validation score to avoid overfitting
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

    if (epoch + 1) % 20 == 0:

        print(f"Epoch {epoch+1}/{epochs}, " f"Loss: {avg_loss:.4f}")

# evaluating MLP
mlp.eval()

results_path = r"E:\MLprojects\maktap-project-1\reports\mlp"
os.makedirs(results_path, exist_ok=True)

thresholds = [0.3, 0.5, 0.7]
all_results = []

with torch.no_grad():
    # validation probabilities
    outputs = mlp(x_validation_tensor)
    proba_mlp = torch.sigmoid(outputs).cpu().numpy().ravel()

    # train probabilities
    outputs_train = mlp(X_train_tensor)
    proba_train_mlp = torch.sigmoid(outputs_train).cpu().numpy().ravel()


for threshold in thresholds:
    ypred_mlp = (proba_mlp >= threshold).astype(int)
    ypred_train_mlp = (proba_train_mlp >= threshold).astype(int)

    # ================================================
    # confusion matrix train
    # ================================================
    cm_train = confusion_matrix(ytrain, ypred_train_mlp)
    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm_train,
        display_labels=["legitimate", "fraud"],
    )
    dsp.plot()
    plt.title(f"MLP train threshold: {threshold}")

    plt.savefig(
        os.path.join(
            results_path,
            f"mlp_cm_train{int(threshold * 10)}.png",
        )
    )
    plt.close()

    # ================================================
    # confusion matrix validation
    # ================================================
    cm = confusion_matrix(yval, ypred_mlp)
    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["legitimate", "fraud"],
    )
    dsp.plot()
    plt.title(f"MLP threshold: {threshold}")

    plt.savefig(
        os.path.join(
            results_path,
            f"mlp_cm{int(threshold * 10)}.png",
        )
    )
    plt.close()

    # ================================================
    # Save train and validation scores to CSV
    # ================================================
    results = pd.DataFrame(
        [
            {
                "model": f"MLP tr = {threshold}",
                "dataset": "train",
                "precision": precision_score(ytrain, ypred_train_mlp),
                "recall": recall_score(ytrain, ypred_train_mlp),
                "f1_score": f1_score(ytrain, ypred_train_mlp),
                "accuracy": accuracy_score(ytrain, ypred_train_mlp),
            },
            {
                "model": f"MLP tr = {threshold}",
                "dataset": "validation",
                "precision": precision_score(yval, ypred_mlp),
                "recall": recall_score(yval, ypred_mlp),
                "f1_score": f1_score(yval, ypred_mlp),
                "accuracy": accuracy_score(yval, ypred_mlp),
            },
        ]
    )

    results.to_csv(
        os.path.join(
            results_path,
            f"mlp_scores{int(threshold * 10)}.csv",
        ),
        index=False,
    )

    all_results.append(
        {
            "threshold": threshold,
            "f1_score": f1_score(yval, ypred_mlp),
        }
    )

    print(f"\nMLP threshold {threshold} scores saved successfully.")
# find best threshold
f1_list = [res["f1_score"] for res in all_results]
best_idx = f1_list.index(max(f1_list))
best_threshold = all_results[best_idx]["threshold"]

print(f"\nBest MLP threshold based on validation F1: {best_threshold}")
# save model

model_path = r"E:\MLprojects\maktap-project-1\models"
os.makedirs(model_path, exist_ok=True)

torch.save(
    {
        "state_dict": mlp.state_dict(),
        "threshold": best_threshold,
        "input_dim": 30,
    },
    os.path.join(model_path, "mlp.pt"),
)
