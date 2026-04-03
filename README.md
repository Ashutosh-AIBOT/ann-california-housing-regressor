---
title: California Housing Regressor
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🏠 California Housing Regressor (ANN Regression)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)

A professional-grade regression pipeline for predicting California house prices based on 1990 Census data. This project uses an Artificial Neural Network (ANN) to model non-linear relationships in housing features such as median income, house age, and spatial location.

---

## 📋 Project Overview

### 1. The Problem
Accurate real estate valuation is a complex non-linear problem. This project provides a robust regression model to estimate median house values, facilitating automated appraisal and market analysis.

### 2. The Dataset
* **Source**: Scikit-learn California Housing Dataset (1990 Census).
* **Target**: Median house price for California districts (measured in hundreds of thousands of dollars).
* **Features**: 8 predictive variables including `MedInc`, `HouseAge`, `AveRooms`, and geographical `Latitude`/`Longitude`.

### 3. The Solution (ANN Architecture)
* **Architecture**: Feed-forward Neural Network (PyTorch).
* **Hidden Layers**:
  * Dense (64 units) → BatchNorm → ReLU → Dropout (0.2)
  * Dense (32 units) → ReLU → Dropout (0.1)
  * Dense (1) → Regression Output.
* **Optimization**: Adam optimizer with MSE (Mean Squared Error) Loss.

---

## 📂 Project Structure

```text
.
├── app.py                # Streamlit Dashboard (UI/UX)
├── dashboard_core.py     # Inference & Metric logic
├── Dockerfile            # Containerization
├── requirements.txt      # Dependencies
├── models/               # Trained ANN model (PyTorch .pth & .pkl)
├── charts/               # Training curves & Performance visualizations
├── data/                 # Processed California Housing dataset
└── notebooks/            # Jupyter Notebooks (EDA, Preprocessing, Training)
    ├── 01_eda.ipynb
    ├── 02_preprocessing.ipynb
    └── 03_train_evaluate.ipynb
```

---


## 📈 Numerical Results & Performance

The model demonstrates excellent convergence and predictive accuracy:

| Metric | Value | Significance |
| :--- | :--- | :--- |
| **R2 Score** | **0.7925** | Explains nearly 80% of the variance in housing prices. |
| **RMSE** | **0.5214** | Root Mean Squared Error (Standard deviation of prediction errors). |
| **MAE** | **0.3479** | Mean Absolute Error (Average magnitude of errors). |
| **Loss (MSE)** | **0.2719** | Optimized MSE on the normalized test set. |

---

## 🚀 Live Deployment

### Dockerized Hosting (HF Spaces / Cloud)
This project is fully containerized for deployment:

1. **Build the Image**:
   ```bash
   docker build -t california-housing-regressor .
   ```
2. **Run the Dashboard**:
   ```bash
   docker run -p 8501:7860 california-housing-regressor
   ```

### Standard Local Execution
1. **Activate Environment**:
   ```bash
   conda activate ml-env
   ```
2. **Launch Streamlit**:
   ```bash
   streamlit run app.py
   ```

---

## 🛠️ GitHub Configuration & Workflow

```bash
# 1. Initialize & track with LFS
git init
git lfs install
git lfs track "models/*.pkl" "charts/*.png"

# 2. Deploy to GitHub
git add .
git commit -m "feat: Initial professional deployment of California Housing regression model and analytics dashboard"
git branch -M main
git remote add origin git@github.com:Ashutosh-AIBOT/ann-california-housing-regressor.git
git push -u origin main
```

---

**Developed by [Ashutosh-AIBOT](https://github.com/Ashutosh-AIBOT)**
