import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from dashboard_core import get_regression_metrics, get_training_curves, predict_price, load_model_and_scaler
from path_utils import CHARTS

st.set_page_config(page_title="California Housing Regressor", layout="wide")
st.title("California Housing Price Predictor")
st.caption("Prediction is in $100k units")
st.caption("Raw input -> StandardScaler -> ANN regressor -> price estimate")

# --- HEALTH CHECK ---
model, scaler = load_model_and_scaler()
with st.sidebar:
    st.header("Housing Features")
    if model and scaler:
        st.success("✅ Model Loaded Successfully")
    else:
        st.error("❌ Model/Scaler not found. Please verify deployment artifacts.")
        st.info("Ensure `models/` and `data/processed/` directories are populated.")

features = {
    "MedInc": st.sidebar.slider("MedInc", 0.5, 15.0, 3.9, 0.1),
    "HouseAge": st.sidebar.slider("HouseAge", 1.0, 55.0, 28.0, 1.0),
    "AveRooms": st.sidebar.slider("AveRooms", 1.0, 15.0, 5.4, 0.1),
    "AveBedrms": st.sidebar.slider("AveBedrms", 0.5, 5.0, 1.1, 0.1),
    "Population": st.sidebar.slider("Population", 3.0, 35000.0, 1400.0, 50.0),
    "AveOccup": st.sidebar.slider("AveOccup", 1.0, 10.0, 3.0, 0.1),
    "Latitude": st.sidebar.slider("Latitude", 32.0, 42.0, 34.0, 0.1),
    "Longitude": st.sidebar.slider("Longitude", -124.0, -114.0, -118.0, 0.1),
}

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if st.button("Predict Price", disabled=(model is None)):
    try:
        pred = predict_price(features)
        st.session_state.prediction_history.append(pred)

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted Price ($100k)", f"{pred:.3f}")
        with c2:
            st.metric("Predicted Price (USD)", f"${pred * 100000:,.0f}")

        st.line_chart(np.array(st.session_state.prediction_history, dtype=float))
        st.caption("Live prediction history (updates every time you click Predict Price).")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

curves_tab, pred_tab, resid_tab, arch_tab = st.tabs(
    ["Training Curve", "Pred vs Actual", "Residuals + Metrics", "Model & Metric Guide"]
)

with curves_tab:
    hist = get_training_curves()
    if hist:
        x = np.arange(1, len(hist["train_loss"]) + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, hist["train_loss"], label="train")
        ax.plot(x, hist["val_loss"], label="val")
        ax.set_title("Loss Curve")
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Run notebook 03 to generate training history.")

with pred_tab:
    path = CHARTS / "pred_vs_actual.png"
    if path.exists():
        st.image(str(path), caption="Predicted vs Actual")
    else:
        st.info("Run notebook 03 to generate predicted-vs-actual chart.")

with resid_tab:
    c1, c2 = st.columns(2)
    with c1:
        p = CHARTS / "residual_hist.png"
        if p.exists():
            st.image(str(p), caption="Residual Histogram")
        else:
            st.info("Residual histogram not found yet.")
    with c2:
        p = CHARTS / "residual_vs_pred.png"
        if p.exists():
            st.image(str(p), caption="Residual vs Predicted")
        else:
            st.info("Residual-vs-predicted chart not found yet.")

    metrics = get_regression_metrics()
    if metrics:
        cols = st.columns(4)
        if "rmse" in metrics:
            cols[0].metric("RMSE", f"{metrics['rmse']:.4f}")
        if "mae" in metrics:
            cols[1].metric("MAE", f"{metrics['mae']:.4f}")
        if "r2" in metrics:
            cols[2].metric("R²", f"{metrics['r2']:.4f}")
        if "explained_variance" in metrics:
            cols[3].metric("Explained Var", f"{metrics['explained_variance']:.4f}")

        extra = {}
        for key in ["mse", "median_ae", "mape", "smape"]:
            if key in metrics:
                extra[key] = metrics[key]
        if extra:
            st.write("Additional regression diagnostics:")
            st.json(extra)

with arch_tab:
    st.code(
        "Input(8)  -> Linear(128) -> BatchNorm1d -> ReLU -> Dropout(0.2)\n"
        "         -> Linear(64)  -> ReLU\n"
        "         -> Linear(32)  -> ReLU\n"
        "         -> Linear(1)   -> output price ($100k)",
        language="text",
    )
    st.markdown("### Metric Guide")
    l, r = st.columns(2)
    with l:
        st.success(
            "**RMSE / MAE**\n\n"
            "Average prediction error in target units.\n"
            "Lower is better."
        )
        st.info(
            "**R² / Explained Variance**\n\n"
            "How much variance in price is explained by the model.\n"
            "Higher is better."
        )
    with r:
        st.warning(
            "**MAPE / SMAPE**\n\n"
            "Percentage-style error. Useful for business interpretation.\n"
            "Lower is better."
        )
        st.info(
            "**Residual Plots**\n\n"
            "Best case is centered around zero with no clear pattern."
        )
