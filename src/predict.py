"""
This file is for predicting :
 a request in JSON format will be received and the answer will be sent in JSON format.
"""

# ==============imports======================
import joblib
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent
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


# # ================requests function===================
def predict(json_request):
    """This function answer in JSON the request that received in JSON"""
    try:
        # ================ json => tensor ===================

        data = json.loads(json_request)
        data_frame = pd.DataFrame([data])
        df_scaled = scaler.transform(data_frame.to_numpy())
        tensor_data = torch.tensor(df_scaled, dtype=torch.float32)
        # ================Extract probability and prediction===================
        with torch.no_grad():
            output = model(tensor_data)
            probability = torch.sigmoid(output).item()
        prediction = "Fraud" if probability >= best_threshold else "Legitimate"
        class_id = 1 if prediction == "Fraud" else 0

        # ================return output===================
        result = {
            "prediction": prediction,
            "class_id": class_id,
            "probability": probability,
            "threshold": best_threshold,
            "status": "success",
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


# ================scaler===================
scaler_path = base_dir/"models"/"scaler.pkl"
scaler = joblib.load(scaler_path)

# ================MLP stats===================
model_path = base_dir/"models"/"bestmodel.pt"
model_stats = torch.load(model_path)
best_threshold = model_stats["threshold"]
# ================MLP obj===================
model = MLP()
model.load_state_dict(model_stats["state_dict"])
model.eval()

# ================test performance===================

if __name__ == "__main__":
    sample_json = json.dumps(
        {
            "Time": 406,
            "V1": -1.359807,
            "V2": -0.072781,
            "V3": 2.536347,
            "V4": 1.378155,
            "V5": -0.338321,
            "V6": 0.462388,
            "V7": 0.239599,
            "V8": 0.098698,
            "V9": 0.363787,
            "V10": 0.090794,
            "V11": -0.551600,
            "V12": -0.617801,
            "V13": -0.991390,
            "V14": -0.311169,
            "V15": 1.468177,
            "V16": -0.470401,
            "V17": 0.207971,
            "V18": 0.025791,
            "V19": 0.403993,
            "V20": 0.251412,
            "V21": -0.018307,
            "V22": 0.277838,
            "V23": -0.110474,
            "V24": 0.066928,
            "V25": 0.128539,
            "V26": -0.189115,
            "V27": 0.133558,
            "V28": -0.021053,
            "Amount": 149.62,
        }
    )
    print(predict(sample_json))
