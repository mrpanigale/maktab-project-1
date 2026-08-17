"""In This file we wanna test our KNN on test data"""

#=============imports==============
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from src import data_prep
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)

base_dir = Path(__file__).resolve().parent.parent.parent

#=============load model and scaler==============
try:
    model_path = base_dir / "models"/"knn.pkl"
    model = joblib.load(model_path)
except FileNotFoundError:
    print("model is not in right path please run knn.py first")
    sys.exit(1)

try:
    scaler_path = base_dir / "models"/"scaler.pkl"
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    print("scaler is not in right path please run data_prep.py first")
    sys.exit(1)

#=============Load-Data==============
Xtest,ytest = data_prep.loader(show_print=False,return_test=True)
Xtest_scaled = scaler.transform(Xtest)
#=============predict==============
threshold = 0.3
yproba =model.predict_proba(Xtest_scaled)
ypred =  yproba[:,1] >= threshold

#=============metrics==============
f1 = f1_score(ytest,ypred,zero_division=0)
recall = recall_score(ytest,ypred,zero_division=0)
precision = precision_score(ytest,ypred,zero_division=0)
accuracy = accuracy_score(ytest,ypred)

result = pd.DataFrame({
                # =============Test==============

                "model": f"KNN tr={threshold}",
                "dataset": "Test",
                "precision": [precision],
                "recall": [recall],
                "f1_score": [f1],
                "accuracy": [accuracy],
                "best_K": [model.get_params()["n_neighbors"]],
            })

result_path = base_dir/"reports"/"final_evaluate"/"knn"
result_path.mkdir(parents=True, exist_ok=True)
result.to_csv(
    result_path / f"knn.csv", index=False
)
#=============confusion matrix==============

cm = confusion_matrix(ytest, ypred)
dsp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["legitimate", "fraud"]
)
dsp.plot()
plt.title(f"KNN Test | C={model.get_params()["n_neighbors"]} | TR={threshold}")
plt.savefig(result_path / "knn_cm.png")
plt.close()
print(f"KNN threshold {threshold} results saved.")