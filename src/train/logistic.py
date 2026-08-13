"""This file trains four model Logistic Regression."""

# imports
import os
import matplotlib.pyplot as plt
import joblib
# ==models==
from src import data_prep
from sklearn.linear_model import LogisticRegression

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


# ============================================================
# load data from data_prep.py
# ============================================================
Xtrain, Xtest_final, ytrain, ytest_final = data_prep.loader(show_print=False)
Xtrain, Xval, ytrain, yval = train_test_split(
    Xtrain, ytrain, test_size=0.2, random_state=42, stratify=ytrain
)

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

# save model

model_path = r"E:\MLprojects\maktap-project-1\models"
os.makedirs(model_path, exist_ok=True)

joblib.dump(logreg.best_estimator_, os.path.join(model_path, "logistic_regression.pkl"))
