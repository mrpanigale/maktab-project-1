"""This file trains Decision Tree model."""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from src import data_prep
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def decision_tree_fitter(Xtrain, Xtest, ytrain, max_depth: list):
    """this function Train Decision Tree."""

    tree = DecisionTreeClassifier(random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    dict_params = {"max_depth": max_depth}

    grid = GridSearchCV(
        estimator=tree, param_grid=dict_params, cv=skf, scoring="f1", n_jobs=-1
    )

    grid.fit(Xtrain, ytrain)

    ypred_proba = grid.predict_proba(Xtest)[:, 1]
    ypred_train_proba = grid.predict_proba(Xtrain)[:, 1]

    return ypred_proba, ypred_train_proba, grid


# ============================================================
# Load data from data_prep.py
# ============================================================
Xtrain, Xval, Xtest_final, ytrain, yval, ytest_final = data_prep.loader(
    show_print=False
)

# we don't need to scale data for tree because it's not sensitive to scale

# ============================================================
# Decision Tree training
# ============================================================
proba_tree, proba_train_tree, tree = decision_tree_fitter(
    Xtrain, Xval, ytrain, max_depth=[2, 5, 10, None]
)

thresholds = [0.3, 0.5, 0.7]

results_path = r"E:\MLprojects\maktap-project-1\reports\tree"
os.makedirs(results_path, exist_ok=True)

for threshold in thresholds:
    threshold_name = int(threshold * 10)

    ypred_tree = (proba_tree >= threshold).astype(int)
    ypred_train_tree = (proba_train_tree >= threshold).astype(int)

    # ========================================================
    # Train confusion matrix
    # ========================================================
    cm_train = confusion_matrix(ytrain, ypred_train_tree)
    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=["legitimate", "fraud"]
    )
    dsp.plot()
    plt.title(f"Tree Train | Depth={tree.best_params_['max_depth']} | TR={threshold}")
    plt.savefig(os.path.join(results_path, f"tree_cm_train_{threshold_name}.png"))
    plt.close()

    # ========================================================
    # Validation confusion matrix
    # ========================================================
    cm = confusion_matrix(yval, ypred_tree)
    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["legitimate", "fraud"]
    )
    dsp.plot()
    plt.title(
        f"Tree Validation | Depth={tree.best_params_['max_depth']} | TR={threshold}"
    )
    plt.savefig(os.path.join(results_path, f"tree_cm_{threshold_name}.png"))
    plt.close()

    # ========================================================
    # Save scores to CSV
    # ========================================================
    results = pd.DataFrame(
        [
            {
                "model": f"Decision Tree tr={threshold}",
                "dataset": "train",
                "precision": precision_score(ytrain, ypred_train_tree),
                "recall": recall_score(ytrain, ypred_train_tree),
                "f1_score": f1_score(ytrain, ypred_train_tree),
                "accuracy": accuracy_score(ytrain, ypred_train_tree),
                "best_depth": tree.best_params_["max_depth"],
            },
            {
                "model": f"Decision Tree tr={threshold}",
                "dataset": "validation",
                "precision": precision_score(yval, ypred_tree),
                "recall": recall_score(yval, ypred_tree),
                "f1_score": f1_score(yval, ypred_tree),
                "accuracy": accuracy_score(yval, ypred_tree),
                "best_depth": tree.best_params_["max_depth"],
            },
        ]
    )

    results.to_csv(
        os.path.join(results_path, f"tree_scores_{threshold_name}.csv"), index=False
    )

    print(f"Decision Tree threshold {threshold} results saved.")

# ============================================================
# Save model
# ============================================================
model_path = r"E:\MLprojects\maktap-project-1\models"
os.makedirs(model_path, exist_ok=True)

joblib.dump(tree.best_estimator_, os.path.join(model_path, "decision_tree.pkl"))
