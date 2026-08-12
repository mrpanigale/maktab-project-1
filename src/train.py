"""This file trains four models: MLP, Decision Tree, KNN, and Logistic Regression."""

import os

# imports
import matplotlib.pyplot as plt
import joblib
# ==models==
import data_prep
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

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


# functions
def knn_fitter(Xtrain, Xtest, ytrain, n_neighbors: list):
    """This function trains KNN classifier."""
    knn = KNeighborsClassifier()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dict_params = {"n_neighbors": n_neighbors}
    grid = GridSearchCV(
        estimator=knn, param_grid=dict_params, cv=skf, scoring="f1", n_jobs=-1
    )
    grid.fit(Xtrain, ytrain)
    y_pred, ypred_train = grid.predict(Xtest), grid.predict(Xtrain)
    return y_pred, ypred_train, grid


def decision_tree_fitter(Xtrain, Xtest, ytrain, max_depth: list):
    """This function trains Decision Tree classifier."""
    tree = DecisionTreeClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dict_params = {"max_depth": max_depth}
    grid = GridSearchCV(
        estimator=tree, param_grid=dict_params, cv=skf, scoring="f1", n_jobs=-1
    )
    grid.fit(Xtrain, ytrain)
    y_pred, ypred_train = grid.predict(Xtest), grid.predict(Xtrain)
    return y_pred, ypred_train, grid


def logistic_regression_fitter(Xtrain, ytrain, Xtest):
    """This function trains Logistic Regression classifier."""
    logreg = LogisticRegression()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=logreg,
        param_grid={"C": [0.01, 0.1, 1, 10]},
        scoring="f1",
        n_jobs=-1,
        cv=skf,
    )
    grid.fit(Xtrain, ytrain)
    ypred, ypred_train = grid.predict(Xtest), grid.predict(Xtrain)
    return ypred, ypred_train, grid


# MLP sructure
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
# ================================================
# KNN training
# ================================================
ypred_knn, ypred_trian_knn, knn = knn_fitter(
    Xtrain, Xval, ytrain, n_neighbors=[1, 5, 20]
)
# print outputs train
print(
f"{"KNN precision train":<20}: {precision_score(ytrain,ypred_trian_knn):.3f}\n\
{"KNN recall train":<20}: {recall_score(ytrain,ypred_trian_knn):.3f}\n\
{"KNN f1 train":<20}: {f1_score(ytrain,ypred_trian_knn):.3f}\n\
{"KNN accuracy train":<20}: {accuracy_score(ytrain,ypred_trian_knn):.3f}"
)

# confusion matrix train
cm_train = confusion_matrix(ytrain, ypred_trian_knn)
dsp = ConfusionMatrixDisplay(
    confusion_matrix=cm_train, display_labels=["legitimate", "fraud"]
)
dsp.plot()
plt.title(f"KNN train cm K = {knn.best_params_['n_neighbors']}")

# save plot

os.makedirs(r"E:\MLprojects\maktap-project-1\reports\knn", exist_ok=True)
plt.savefig(r"E:\MLprojects\maktap-project-1\reports\knn\knn_cm_train.png")
plt.close()
# ================================= test
# outputs test
print(
f"{"KNN precision":<20}: {precision_score(yval,ypred_knn):.3f}\n\
{"KNN recall":<20}: {recall_score(yval,ypred_knn):.3f}\n\
{"KNN f1":<20}: {f1_score(yval,ypred_knn):.3f}\n\
{"KNN accuracy":<20}: {accuracy_score(yval,ypred_knn):.3f}"
)

# confusion matrix test
cm = confusion_matrix(yval, ypred_knn)
dsp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["legitimate", "fraud"]
)
dsp.plot()
plt.title(f"KNN cm K = {knn.best_params_['n_neighbors']}")

# save plot test
os.makedirs(r"E:\MLprojects\maktap-project-1\reports\knn", exist_ok=True)
plt.savefig(r"E:\MLprojects\maktap-project-1\reports\knn\knn_cm.png")
plt.close()


# ================================================
# Decision Tree training
# ================================================
ypred_tree, ypred_trian_tree, tree = decision_tree_fitter(
    Xtrain, Xval, ytrain, max_depth=[2, 5, 10, None]
)

# print outputs train
print(
    f"{'Tree precision train':<20}: {precision_score(ytrain, ypred_trian_tree):.3f}\n\
{'Tree recall train':<20}: {recall_score(ytrain, ypred_trian_tree):.3f}\n\
{'Tree f1 train':<20}: {f1_score(ytrain, ypred_trian_tree):.3f}\n\
{'Tree accuracy train':<20}: {accuracy_score(ytrain, ypred_trian_tree):.3f}"
)
print("-"*30)

# confusion matrix train
cm_train = confusion_matrix(ytrain, ypred_trian_tree)
dsp = ConfusionMatrixDisplay(
    confusion_matrix=cm_train, display_labels=["legitimate", "fraud"]
)
dsp.plot()
plt.title(f"Decision Tree train depth = {tree.best_params_['max_depth']}")

# save plot
os.makedirs(r"E:\MLprojects\maktap-project-1\reports\tree", exist_ok=True)
plt.savefig(r"E:\MLprojects\maktap-project-1\reports\tree\tree_cm_train.png")
plt.close()

# ================================= test
# outputs test
print(
    f"{'Tree precision':<20}: {precision_score(yval, ypred_tree):.3f}\n\
{'Tree recall':<20}: {recall_score(yval, ypred_tree):.3f}\n\
{'Tree f1':<20}: {f1_score(yval, ypred_tree):.3f}\n\
{'Tree accuracy':<20}: {accuracy_score(yval, ypred_tree):.3f}"
)
print("-"*30)

# confusion matrix test
cm = confusion_matrix(yval, ypred_tree)
dsp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["legitimate", "fraud"]
)
dsp.plot()
plt.title(f"Decision Tree cm depth = {tree.best_params_['max_depth']}")

# save plot test
os.makedirs(r"E:\MLprojects\maktap-project-1\reports\tree", exist_ok=True)
plt.savefig(r"E:\MLprojects\maktap-project-1\reports\tree\tree_cm.png")
plt.close()


# ================================================
# Logistic Regression training
# ================================================
ypred_logreg, ypred_trian_logreg, logreg = logistic_regression_fitter(
    Xtrain, ytrain, Xval
)

# print outputs train
print(
f"{'LogReg precision train':<20}: {precision_score(ytrain, ypred_trian_logreg):.3f}\n\
{'LogReg recall train':<20}: {recall_score(ytrain, ypred_trian_logreg):.3f}\n\
{'LogReg f1 train':<20}: {f1_score(ytrain, ypred_trian_logreg):.3f}\n\
{'LogReg accuracy train':<20}: {accuracy_score(ytrain, ypred_trian_logreg):.3f}"
)
print("-"*30)
# confusion matrix train
cm_train = confusion_matrix(ytrain, ypred_trian_logreg)
dsp = ConfusionMatrixDisplay(
    confusion_matrix=cm_train, display_labels=["legitimate", "fraud"]
)
dsp.plot()
plt.title(f"LogReg train C = {logreg.best_params_['C']}")

# save plot
os.makedirs(r"E:\MLprojects\maktap-project-1\reports\logreg", exist_ok=True)
plt.savefig(r"E:\MLprojects\maktap-project-1\reports\logreg\logreg_cm_train.png")
plt.close()

# ================================= test
# outputs test
print(
f"{'LogReg precision':<20}: {precision_score(yval, ypred_logreg):.3f}\n\
{'LogReg recall':<20}: {recall_score(yval, ypred_logreg):.3f}\n\
{'LogReg f1':<20}: {f1_score(yval, ypred_logreg):.3f}\n\
{'LogReg accuracy':<20}: {accuracy_score(yval, ypred_logreg):.3f}"
)

# confusion matrix test
cm = confusion_matrix(yval, ypred_logreg)
dsp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["legitimate", "fraud"]
)
dsp.plot()
plt.title(f"LogReg cm C = {logreg.best_params_['C']}")

# save plot test
os.makedirs(r"E:\MLprojects\maktap-project-1\reports\logreg", exist_ok=True)
plt.savefig(r"E:\MLprojects\maktap-project-1\reports\logreg\logreg_cm.png")
plt.close()


# save models

model_path = r"E:\MLprojects\maktap-project-1\models"
os.makedirs(model_path, exist_ok=True)

joblib.dump(knn.best_estimator_, os.path.join(model_path, "knn.pkl"))
joblib.dump(tree.best_estimator_, os.path.join(model_path, "decision_tree.pkl"))
joblib.dump(logreg.best_estimator_, os.path.join(model_path, "logistic_regression.pkl"))

torch.save(
    {
        "state_dict": mlp.state_dict(),
        "threshold": 0.3,
        "input_dim": 30,
    },
    os.path.join(model_path, "mlp.pt"),
)


