"""This file trains four models: MLP and Logistic Regression."""


# imports
import os
import matplotlib.pyplot as plt
import joblib
from src import data_prep

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


# MLP structure
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
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
Xtrain, Xtest_final, ytrain, ytest_final = data_prep.loader(show_print=False)
Xtrain, Xval, ytrain, yval = train_test_split(
    Xtrain, ytrain, test_size=0.2, random_state=42, stratify=ytrain
)
# tensor X,y
X_train_tensor = torch.tensor(Xtrain, dtype=torch.float32)
x_validation_tensor = torch.tensor(Xval, dtype=torch.float32)
y_train_tensor = torch.tensor(ytrain, dtype=torch.float32).reshape(-1, 1)
y_test_tensor = torch.tensor(yval, dtype=torch.float32).reshape(-1, 1)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# return batchs
train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

# training loop
epochs = 40

loss_history = []
loss_history_batch = []

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

    loss_history.append(avg_loss)

    if (epoch + 1) % 20 == 0:

        print(f"Epoch {epoch+1}/{epochs}, " f"Loss: {avg_loss:.4f}")

# evaluating MLP
mlp.eval()

with torch.no_grad():
    # output validation
    outputs = mlp(x_validation_tensor)
    # treshold
    threshold = 0.3
    # apply step function on probability
    ypred_mlp = (torch.sigmoid(outputs) >= threshold).float().cpu().numpy()
    ytrue_mlp = y_test_tensor.cpu().numpy()
    # outputs

    print(
f"{"MLP precision":<20}: {precision_score(ytrue_mlp,ypred_mlp):.3f}\n\
{"MLP recall":<20}: {recall_score(ytrue_mlp,ypred_mlp):.3f}\n\
{"MLP f1 score":<20}: {f1_score(ytrue_mlp,ypred_mlp):.3f}\n\
{"MLP accuracy":<20}: {accuracy_score(ytrue_mlp,ypred_mlp):.3f}"
    )
    print("-"*30)
    # confusion matrix test
    cm = confusion_matrix(ytrue_mlp, ypred_mlp)
    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["legitimate", "fraud"]
    )
    dsp.plot()
    plt.title("MLP")
    # save plot

    os.makedirs(r"E:\MLprojects\maktap-project-1\reports\mlp", exist_ok=True)
    plt.savefig(r"E:\MLprojects\maktap-project-1\reports\mlp\mlp_cm.png")
    plt.close()
    # ================== train
    # output train
    outputs_train = mlp(X_train_tensor)

    # apply step function on probability train
    ypred_train_mlp = (torch.sigmoid(outputs_train) >= threshold).float().cpu().numpy()
    ytrue_train_mlp = y_train_tensor.cpu().numpy()

    # scores outputs train

    print(
f"{"MLP precision train":<20}: {precision_score(ytrue_train_mlp,ypred_train_mlp):.3f}\n\
{"MLP recall train":<20}: {recall_score(ytrue_train_mlp,ypred_train_mlp):.3f}\n\
{"MLP f1 score train":<20}: {f1_score(ytrue_train_mlp,ypred_train_mlp):.3f}\n\
{"MLP accuracy train":<20}: {accuracy_score(ytrue_train_mlp,ypred_train_mlp):.3f}\n"
)
    print("-"*30)
    # confusion matrix
    cm_train = confusion_matrix(ytrue_train_mlp, ypred_train_mlp)
    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=["legitimate", "fraud"]
    )
    dsp.plot()
    plt.title("MLP train")

    # save trian plot

    os.makedirs(r"E:\MLprojects\maktap-project-1\reports\mlp", exist_ok=True)
    plt.savefig(r"E:\MLprojects\maktap-project-1\reports\mlp\mlp_cm_train.png")
    plt.close()

# save model

model_path = r"E:\MLprojects\maktap-project-1\models"
os.makedirs(model_path, exist_ok=True)

torch.save(
    {
        "state_dict": mlp.state_dict(),
        "threshold": 0.3,
        "input_dim": 30,
    },
    os.path.join(model_path, "mlp.pt"),
)


