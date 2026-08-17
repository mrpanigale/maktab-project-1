"""This file trains Logistic Regression model."""

# =============imports==============
import joblib
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from src import data_prep
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

base_dir = Path(__file__).resolve().parent.parent.parent
# =============logreg trainer==============
def logistic_regression_fitter(Xtrain, Xtest, ytrain):
    """This function trains logistic regression."""

    # =============model==============
    logreg = LogisticRegression(max_iter=1000,random_state=42)

    # =============Cross-Validation==============
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    grid = GridSearchCV(
        estimator=logreg,
        param_grid={"C": [0.01, 0.1, 1, 10]},
        scoring="f1",
        n_jobs=-1,
        cv=skf,
    )

    grid.fit(Xtrain, ytrain)

    # =============Extract-probability==============
    ypred_proba = grid.predict_proba(Xtest)[:, 1]
    ypred_train_proba = grid.predict_proba(Xtrain)[:, 1]

    return ypred_proba, ypred_train_proba, grid


# ============================================================
#             Load data from data_prep.py
# ============================================================
Xtrain, Xval ,ytrain, yval = data_prep.loader(
    show_print=False,
    return_test=False
)

# =============Scaler==============
try:
    scaler_path = base_dir/"models"/"scaler.pkl"
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    print("scaler is not in right path please run data_prep.py first")
    sys.exit(1)


Xtrain_scaled = scaler.transform(Xtrain)
Xval_scaled = scaler.transform(Xval)

# ============================================================
#            Logistic Regression training
# ============================================================
proba_logreg, proba_train_logreg, logreg = logistic_regression_fitter(
    Xtrain_scaled,
    Xval_scaled,
    ytrain,
)

thresholds = [0.3, 0.5, 0.7]

# =============Create and Check report path==============
results_path = base_dir/"reports"/"logreg"
results_path.mkdir(parents=True, exist_ok=True)
# =============Search-Thresholds==============
for threshold in thresholds:
    threshold_name = int(threshold * 10)

    ypred_logreg = (proba_logreg >= threshold).astype(int)
    ypred_train_logreg = (proba_train_logreg >= threshold).astype(int)

    # ========================================================
    #              Train confusion matrix
    # ========================================================
    cm_train = confusion_matrix(ytrain, ypred_train_logreg)

    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm_train,
        display_labels=["legitimate", "fraud"],
    )
    dsp.plot()

    plt.title(f"LogReg train | C={logreg.best_params_['C']} | threshold={threshold}")
    plt.savefig(
        results_path/f"logreg_cm_train_{threshold_name}.png"
    )
    plt.close()

    # ========================================================
    #              Validation confusion matrix
    # ========================================================
    cm = confusion_matrix(yval, ypred_logreg)

    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["legitimate", "fraud"],
    )
    dsp.plot()

    plt.title(
        f"LogReg validation | C={logreg.best_params_['C']} | threshold={threshold}"
    )
    plt.savefig(
        results_path/f"logreg_cm_{threshold_name}.png"
    )
    plt.close()

    # ========================================================
    #       Save train and validation scores to CSV
    # ========================================================

    results = pd.DataFrame(
        [
            {
                # =============train==============
                "model": f"Logistic Regression scaled tr = {threshold}",
                "dataset": "train",
                "precision": precision_score(ytrain, ypred_train_logreg,zero_division=0),
                "recall": recall_score(ytrain, ypred_train_logreg,zero_division=0),
                "f1_score": f1_score(ytrain, ypred_train_logreg,zero_division=0),
                "accuracy": accuracy_score(ytrain, ypred_train_logreg),
                "best_C": logreg.best_params_["C"],
            },
            {
                # =============validation==============
                "model": f"Logistic Regression scaled tr = {threshold}",
                "dataset": "validation",
                "precision": precision_score(yval, ypred_logreg,zero_division=0),
                "recall": recall_score(yval, ypred_logreg,zero_division=0),
                "f1_score": f1_score(yval, ypred_logreg,zero_division=0),
                "accuracy": accuracy_score(yval, ypred_logreg),
                "best_C": logreg.best_params_["C"],
            },
        ]
    )

    # =============Save-reports==============
    results.to_csv(
        results_path/f"logreg_scores_{threshold_name}.csv",
        index=False,
    )

    print(f"\nLogistic Regression threshold {threshold} saved successfully.")

# ============================================================
#                    Save model
# ============================================================
model_path = base_dir/"models"
model_path.mkdir(parents=True, exist_ok=True)
joblib.dump(
    logreg.best_estimator_,
    model_path/"logistic_regression.pkl"
)
