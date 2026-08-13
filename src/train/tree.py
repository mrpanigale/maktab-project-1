"""This file trains four model Decision Tree"""

# imports
import os
import matplotlib.pyplot as plt
import joblib
# ==model==
from src import data_prep
from sklearn.tree import DecisionTreeClassifier

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

# ============================================================
# load data from data_prep.py
# ============================================================
Xtrain, Xtest_final, ytrain, ytest_final = data_prep.loader(show_print=False)
Xtrain, Xval, ytrain, yval = train_test_split(
    Xtrain, ytrain, test_size=0.2, random_state=42, stratify=ytrain
)

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


#save model


model_path = r"E:\MLprojects\maktap-project-1\models"
os.makedirs(model_path, exist_ok=True)

joblib.dump(tree.best_estimator_, os.path.join(model_path, "decision_tree.pkl"))