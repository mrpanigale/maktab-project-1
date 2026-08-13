"""This file trains four model KNN"""


# imports
import matplotlib.pyplot as plt
import joblib
import os
from src import data_prep

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
    y_pred, ypred_train = grid.predict(Xtest), grid.predict(Xtrain)
    return y_pred, ypred_train, grid

# ============================================================
# load data from data_prep.py
# ============================================================
Xtrain, Xtest_final, ytrain, ytest_final = data_prep.loader(show_print=False)
Xtrain, Xval, ytrain, yval = train_test_split(
    Xtrain, ytrain, test_size=0.2, random_state=42, stratify=ytrain
)




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


# save model
model_path = r"E:\MLprojects\maktap-project-1\models"
os.makedirs(model_path, exist_ok=True)
joblib.dump(knn.best_estimator_, os.path.join(model_path, "knn.pkl"))
