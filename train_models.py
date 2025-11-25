import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("✅ Starting model training pipeline...")

# --- Step 1: Load dataset ---
print("🔹 Loading dataset...")
df = pd.read_csv("era5_features.csv", parse_dates=["time"])
print(f"✅ Loaded {len(df)} rows.")

# --- Step 2: Define features & target ---
print("🔹 Preparing features and target...")
X = df[["windspeed", "temperature_C", "pressure_hPa",
        "windspeed_roll24h", "temperature_roll7d",
        "windspeed_cubed", "windspeed_lag1"]].fillna(0)
y = df["windspeed"]

# --- NaN check ---
print("🔍 Checking for NaNs in training data...")
print("X NaNs:", X.isnull().sum().sum())
print("y NaNs:", y.isnull().sum())

# --- Step 3: Train/test split ---
print("🔹 Splitting into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
print(f"✅ Training size: {len(X_train)}, Test size: {len(X_test)}")

# --- Step 4: Scale features for SVR ---
print("🔹 Scaling features for SVR...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved as scaler.pkl")

# --- Step 5: Define models ---
models = {
    "SVR": SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42
    )
}

results = []

# --- Step 6: Train, evaluate, and save models ---
for name, model in models.items():
    print(f"\n🚀 Training {name}...")

    if name == "SVR":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    print(f"✅ {name} training complete.")

    # --- Metrics ---
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # --- Weighted MAPE ---
    mask = y_test >= 1.0
    numerator = np.sum(np.abs(y_test[mask] - y_pred[mask]))
    denominator = np.sum(np.abs(y_test[mask]))
    wmape = (numerator / denominator) * 100 if denominator != 0 else np.nan

    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "wMAPE (%)": wmape, "R²": r2})
    joblib.dump(model, f"{name}_model.pkl")
    print(f"✅ {name} model saved as {name}_model.pkl")

# --- Step 7: Print summary table ---
results_df = pd.DataFrame(results)
print("\n📊 Model Comparison Summary")
print(results_df)
print("\n✅ All models trained and saved successfully.")