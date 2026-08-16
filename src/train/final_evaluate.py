"""In This file I'm Going to test my best and choosen model(MLP) on final test data and save the report"""

# ============imports============
import joblib
import os

import pandas as pd
import torch
import torch.nn as nn
from src import data_prep
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt

# ================MLP===================


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


# ================MLP stats===================
model_stats = torch.load(r"models\mlp.pt")
# best_threshold = model_stats["threshold"]
best_threshold = 0.3
# ================MLP obj===================
model = MLP()
model.load_state_dict(model_stats["state_dict"])
# ==========scaler============
scaler = joblib.load("models/scaler.pkl")

Xtrain, Xval, Xtest_final, ytrain, yval, ytest_final = data_prep.loader(
    show_print=False
)
Xtest_final_scaled = scaler.transform(Xtest_final)
Xtest_tensor = torch.tensor(Xtest_final_scaled, dtype=torch.float32)
# ==========predict================

model.eval()
with torch.no_grad():
    output = model(Xtest_tensor)
    probability = torch.sigmoid(output).squeeze().numpy()
    ypred = (probability >= best_threshold).astype(int)

# ================================================
# results path
# ================================================
results_path = r"E:\MLprojects\maktap-project-1\reports\final_evaluate"
os.makedirs(results_path, exist_ok=True)


# ================================================
# confusion matrix final test
# ================================================
cm = confusion_matrix(ytest_final, ypred)
dsp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["legitimate", "fraud"],
)
dsp.plot()
plt.title(f"MLP threshold: {best_threshold}")

plt.savefig(
    os.path.join(
        results_path,
        f"best_cm{int(best_threshold * 10)}.png",
    )
)
plt.close()

# ================================================
# Save final test scores to CSV
# ================================================
results = pd.DataFrame(
    [
        {
            "model": f"MLP tr = {best_threshold}",
            "dataset": "final test",
            "precision": precision_score(ytest_final, ypred, zero_division=0),
            "recall": recall_score(ytest_final, ypred, zero_division=0),
            "f1_score": f1_score(ytest_final, ypred, zero_division=0),
            "accuracy": accuracy_score(ytest_final, ypred),
        }
    ]
)

results.to_csv(
    os.path.join(
        results_path,
        f"best_scores{int(best_threshold * 10)}.csv",
    ),
    index=False,
)



print(f"\n best model scores saved successfully.")

# save model

model_path = r"E:\MLprojects\maktap-project-1\models"
os.makedirs(model_path, exist_ok=True)

torch.save(
    {
        "state_dict": model.state_dict(),
        "threshold": best_threshold,
        "input_dim": 30,
    },
    os.path.join(model_path, "bestmodel.pt"),
)