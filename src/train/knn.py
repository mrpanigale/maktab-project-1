"""This file trains four model KNN"""

# =============imports==============
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import sys
from src import data_prep
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# =============metrics==============
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

base_dir = Path(__file__).resolve().parent.parent.parent
# =============function==============
def knn_fitter(Xtrain, Xtest, ytrain, n_neighbors: list):
    """This function trains KNN classifier."""

    # =============model==============
    knn = KNeighborsClassifier()
    # =============Cross-Validation==============
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    dict_params = {"n_neighbors": n_neighbors}
    grid = GridSearchCV(
        estimator=knn, param_grid=dict_params, cv=skf, scoring="f1", n_jobs=-1
    )
    grid.fit(Xtrain, ytrain)
    # =============Extract-Proba==============
    y_pred_proba, ypred_train_proba = (
        grid.predict_proba(Xtest)[:, 1],
        grid.predict_proba(Xtrain)[:, 1],
    )
    return y_pred_proba, ypred_train_proba, grid


# ============================================================
#               load data from data_prep.py
# ============================================================
# not scaled data
Xtrain, Xval, ytrain, yval = data_prep.loader(
    show_print=False,
    return_test=False
)


# =============Load-Scaler==============
try:
    scaler_path = base_dir/"models"/"scaler.pkl"
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    print("scaler is not in right path please run data_prep.py first")
    sys.exit(1)

Xtrain_scaled = scaler.transform(Xtrain)
Xval_scaled = scaler.transform(Xval)

# ================================================
#        KNN-Scaled training
# ================================================
thresholds = [0.3, 0.5, 0.7]
proba_knn, proba_train_knn, knn = knn_fitter(
    Xtrain_scaled, Xval_scaled, ytrain, n_neighbors=[1, 5, 20]
)

# =============Create and Check path of report directory==============
results_path = base_dir / "reports"/"knn"
results_path.mkdir(exist_ok=True,parents=True)

# =============Search-Thresholds==============
for threshold in thresholds:
    ypred_knn = proba_knn >= threshold
    ypred_train_knn = proba_train_knn >= threshold

    # =============Confusion-Matrix==============
    cm_train = confusion_matrix(ytrain, ypred_train_knn)
    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=["legitimate", "fraud"]
    )
    dsp.plot()
    plt.title(
        f"KNN train cm K = {knn.best_params_['n_neighbors']} threshold: {threshold}"
    )

    # =============Save-Plot==============
    plt.savefig(
        results_path/f"knn_cm_train{int(threshold*10)}.png",
    )
    plt.close()

    # =============Confusion-Test==============
    cm = confusion_matrix(yval, ypred_knn)
    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["legitimate", "fraud"]
    )
    dsp.plot()
    plt.title(f"KNN cm K = {knn.best_params_['n_neighbors']} threshold {threshold}")

    # =============Save-cm==============
    plt.savefig(
        results_path/f"knn_cm{int(threshold*10)}.png"
    )
    plt.close()

    # ================================================
    #    Save train and validation scores to CSV
    # ================================================
    results = pd.DataFrame(
        [
            {
                # =============train==============
                "model": f"KNN scaled tr = {threshold}",
                "dataset": "train",
                "precision": precision_score(ytrain, ypred_train_knn,zero_division=0),
                "recall": recall_score(ytrain, ypred_train_knn,zero_division=0),
                "f1_score": f1_score(ytrain, ypred_train_knn,zero_division=0),
                "accuracy": accuracy_score(ytrain, ypred_train_knn),
                "best_n_neighbors": knn.best_params_["n_neighbors"],
            },
            {
                # =============validation==============
                "model": f"KNN scaled tr = {threshold}",
                "dataset": "validation",
                "precision": precision_score(yval, ypred_knn,zero_division=0),
                "recall": recall_score(yval, ypred_knn,zero_division=0),
                "f1_score": f1_score(yval, ypred_knn,zero_division=0),
                "accuracy": accuracy_score(yval, ypred_knn),
                "best_n_neighbors": knn.best_params_["n_neighbors"],
            },
        ]
    )

    # =============Save-Reports==============
    results.to_csv(
        results_path/f"knn_scores{int(threshold*10)}.csv",
        index=False,
    )

    print("\nKNN scores saved successfully.")


# =============Save-Model==============

model_path = base_dir/"models"
model_path.mkdir(exist_ok=True,parents=True)
joblib.dump(knn.best_estimator_, model_path/ "knn.pkl")
