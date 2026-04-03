from __future__ import annotations

import json
import pickle

import numpy as np
import torch
import torch.nn as nn

from path_utils import DATA_PROCESSED, MODELS


class HouseANN(nn.Module):
    def __init__(self, input_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"]
_cache = {}


def load_model_and_scaler() -> tuple[HouseANN | None, any | None]:
    if _cache:
        return _cache.get("model"), _cache.get("scaler")

    try:
        scaler_path = DATA_PROCESSED / "scaler.pkl"
        if not scaler_path.exists():
            print(f"❌ Error: {scaler_path} not found.")
            return None, None
            
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        model_path = MODELS / "model.pkl"
        if not model_path.exists():
            # Try to fall back to checkpoint_best.pth if model.pkl is missing
            model_path = MODELS / "checkpoint_best.pth"
            
        if not model_path.exists():
             print(f"❌ Error: Model file not found in {MODELS}.")
             return None, None

        model = HouseANN(input_dim=8)
        # Load state dict and handle potential mapping or architecture mismatch
        state_dict = torch.load(model_path, map_location="cpu")
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
            
        model.load_state_dict(state_dict)
        model.eval()

        _cache["model"] = model
        _cache["scaler"] = scaler
        return model, scaler
    except Exception as e:
        print(f"❌ Unexpected error loading model/scaler: {e}")
        return None, None


def predict_price(features_dict: dict) -> float:
    missing = [k for k in FEATURES if k not in features_dict]
    if missing:
        raise ValueError(f"Missing features: {missing}")

    model, scaler = load_model_and_scaler()
    x = np.array([[features_dict[k] for k in FEATURES]], dtype=np.float32)
    x = scaler.transform(x)
    x_t = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        pred = model(x_t).squeeze().item()
    return float(pred)


def get_training_curves() -> dict:
    path = DATA_PROCESSED / "training_history.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def get_regression_metrics() -> dict:
    path = MODELS / "results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def get_residual_data():
    path = DATA_PROCESSED / "residual_data.npz"
    if not path.exists():
        return None, None
    arr = np.load(path)
    return arr["y_true"], arr["y_pred"]
