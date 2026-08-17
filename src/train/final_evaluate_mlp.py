"""In This file I'm Going to test my best and choosen model(MLP) on final test data and save the report"""

# ============imports============
import joblib
from pathlib import Path
import pandas as pd
import torch
import sys
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

base_dir = Path(__file__).resolve().parent.parent.parent
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
model_stats = torch.load(base_dir/"models"/"mlp.pt")
best_threshold = model_stats["threshold"]
# ================MLP obj===================
model = MLP()
model.load_state_dict(model_stats["state_dict"])
# ==========scaler============
try:
    scaler_path = base_dir/"models"/"scaler.pkl"
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    print("scaler is not in right path please run data_prep.py first")
    sys.exit(1)

Xtest_final ,ytest_final = data_prep.loader(
    show_print=False,
    return_test=True
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

results_path = base_dir/"reports"/"final_evaluate"/"mlp"
results_path.mkdir(parents=True, exist_ok=True)


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

plt.savefig(results_path / f"best_cm{int(best_threshold * 10)}.png")
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
    results_path / f"best_scores{int(best_threshold * 10)}.csv",
    index=False,
)



print(f"\n best model scores saved successfully.")

# save model

model_path = base_dir/"models"
model_path.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "state_dict": model.state_dict(),
        "threshold": best_threshold,
        "input_dim": 30,
    },
    model_path / "bestmodel.pt",
)