"""This file trains four model KNN"""

# imports
import matplotlib.pyplot as plt
import joblib
import os
from src import data_prep
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

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
    y_pred_proba, ypred_train_proba = (
        grid.predict_proba(Xtest)[:, 1],
        grid.predict_proba(Xtrain)[:, 1],
    )
    return y_pred_proba, ypred_train_proba, grid


# ============================================================
# load data from data_prep.py
# ============================================================
# not scaled data
Xtrain, Xval, Xtest_final, ytrain, yval, ytest_final = data_prep.loader(
    show_print=False
)


#We will not scale dataset this time to see not scaled data how affect KNN



# ================================================
# KNN training | scaled data
# ================================================
thresholds = [0.3, 0.5, 0.7]
proba_knn, proba_train_knn, knn = knn_fitter(
    Xtrain, Xval, ytrain, n_neighbors=[1, 5, 20]
)

results_path = r"E:\MLprojects\maktap-project-1\reports\knn_not_scaled"
os.makedirs(results_path, exist_ok=True)

for threshold in thresholds:
    ypred_knn = proba_knn >= threshold
    ypred_train_knn = proba_train_knn >= threshold

    # confusion matrix train
    cm_train = confusion_matrix(ytrain, ypred_train_knn)
    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=["legitimate", "fraud"]
    )
    dsp.plot()
    plt.title(
        f"KNN train cm K = {knn.best_params_['n_neighbors']} threshold: {threshold}"
    )

    # save plot

    plt.savefig(
        f"E:\\MLprojects\\maktap-project-1\\reports\\knn_not_scaled\\knn_cm_train{int(threshold*10)}.png"
    )
    plt.close()
    # ================================= test

    # confusion matrix test
    cm = confusion_matrix(yval, ypred_knn)
    dsp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["legitimate", "fraud"]
    )
    dsp.plot()
    plt.title(f"KNN cm K = {knn.best_params_['n_neighbors']} threshold {threshold}")

    # save plot test
    plt.savefig(
        f"E:\\MLprojects\\maktap-project-1\\reports\\knn_not_scaled\\knn_cm{int(threshold*10)}.png"
    )
    plt.close()

    # ================================================
    # Save train and validation scores to CSV
    # ================================================
    results = pd.DataFrame(
        [
            {
                "model": f"KNN scaled tr = {threshold}",
                "dataset": "train",
                "precision": precision_score(ytrain, ypred_train_knn),
                "recall": recall_score(ytrain, ypred_train_knn),
                "f1_score": f1_score(ytrain, ypred_train_knn),
                "accuracy": accuracy_score(ytrain, ypred_train_knn),
                "best_n_neighbors": knn.best_params_["n_neighbors"],
            },
            {
                "model": f"KNN scaled tr = {threshold}",
                "dataset": "validation",
                "precision": precision_score(yval, ypred_knn),
                "recall": recall_score(yval, ypred_knn),
                "f1_score": f1_score(yval, ypred_knn),
                "accuracy": accuracy_score(yval, ypred_knn),
                "best_n_neighbors": knn.best_params_["n_neighbors"],
            },
        ]
    )

    results.to_csv(
        os.path.join(results_path, f"knn_scores{int(threshold*10)}.csv"),
        index=False,
    )

    print("\nKNN scores saved successfully.")


# save model
model_path = r"E:\MLprojects\maktap-project-1\models"
os.makedirs(model_path, exist_ok=True)
joblib.dump(knn.best_estimator_, os.path.join(model_path, f"knn_not_scaled.pkl"))
