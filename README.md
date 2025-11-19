# Learning the Wind: Regression-Based Forecasting for Turbine Performance

## 📖 Project Overview
This repository contains code, datasets, and workflows for my dissertation research on machine learning-based windspeed forecasting. The aim is to model turbine performance using regression methods.

## ⚙️ Methods
- **Data Acquisition**: UK Met Office, OpenWeatherMap, ERA5 datasets
- **Preprocessing**: Handle missing values, normalize variables, remove outliers
- **Models**: SVR, Random Forest, Gradient Boosting (Scikit-learn & XGBoost)
- **Evaluation**: RMSE, MAE, R² metrics

## 📂 Repository Structure
- `data/` → raw and processed datasets (ignored via `.gitignore`)
- `notebooks/` → Jupyter notebooks for experiments
- `src/` → Python scripts for preprocessing and modeling
- `results/` → evaluation outputs and plots
