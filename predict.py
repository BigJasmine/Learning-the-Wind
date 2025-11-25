import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# --- Step 1: Define models to load ---
model_files = {
    "SVR": "SVR_model.pkl",
    "RandomForest": "RandomForest_model.pkl",
    "GradientBoosting": "GradientBoosting_model.pkl"
}

# --- Step 2: Load scaler (needed for SVR only) ---
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = None

# --- Step 3: Load new data for prediction ---
new_data = pd.read_csv("new_era5_features.csv", parse_dates=["time"])

# Select the same features used in training
X_new = new_data[["windspeed", "temperature_C", "pressure_hPa",
                  "windspeed_roll24h", "temperature_roll7d",
                  "windspeed_cubed", "windspeed_lag1"]].fillna(0)

y_true = new_data["windspeed"]

results = []

# --- Step 4: Create output folders ---
os.makedirs("plots", exist_ok=True)

# --- Step 5: Create PDF report ---
with PdfPages("model_report.pdf") as pdf:

    for name, file in model_files.items():
        print(f"\n🔹 Running predictions with {name}...")

        # Load model
        model = joblib.load(file)

        # Scale if SVR
        if name == "SVR" and scaler is not None:
            X_input = scaler.transform(X_new)
        else:
            X_input = X_new

        # Predict
        predictions = model.predict(X_input)

        # Add predictions & residuals
        new_data[f"{name}_Predicted"] = predictions
        new_data[f"{name}_Residuals"] = y_true - predictions

        # --- Metrics ---
        rmse = mean_squared_error(y_true, predictions, squared=False)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        # Weighted MAPE (masking low values < 1 m/s)
        mask = y_true >= 1.0
        numerator = np.sum(np.abs(y_true[mask] - predictions[mask]))
        denominator = np.sum(np.abs(y_true[mask]))
        wmape = (numerator / denominator) * 100 if denominator != 0 else np.nan

        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2, "wMAPE (%)": wmape})

        # --- Plot actual vs predicted ---
        plt.figure(figsize=(12,6))
        plt.plot(new_data["time"], y_true, label="Actual Windspeed", color="blue")
        plt.plot(new_data["time"], predictions, label=f"{name} Predicted", color="red", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Windspeed (m/s)")
        plt.title(f"Actual vs Predicted Windspeed ({name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{name}_actual_vs_predicted.png")
        pdf.savefig()  # Save to PDF
        plt.close()

        # --- Plot residuals over time ---
        plt.figure(figsize=(12,6))
        plt.plot(new_data["time"], new_data[f"{name}_Residuals"], label="Residuals", color="green")
        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel("Residual (m/s)")
        plt.title(f"Residuals Over Time ({name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{name}_residuals_over_time.png")
        pdf.savefig()
        plt.close()

        # --- Residual distribution ---
        plt.figure(figsize=(8,6))
        plt.hist(new_data[f"{name}_Residuals"], bins=30, color="purple", alpha=0.7)
        plt.xlabel("Residual (m/s)")
        plt.ylabel("Frequency")
        plt.title(f"Residual Distribution ({name})")
        plt.tight_layout()
        plt.savefig(f"plots/{name}_residual_distribution.png")
        pdf.savefig()
        plt.close()

    # --- Step 6: Add metrics summary table to PDF ---
    results_df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8,3))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=results_df.values,
                     colLabels=results_df.columns,
                     loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    plt.title("📊 Model Comparison Summary")
    pdf.savefig()
    plt.close()

# --- Step 7: Save combined predictions & metrics ---
new_data.to_csv("predictions_all_models.csv", index=False)
results_df.to_csv("model_metrics_summary.csv", index=False)

print("\n✅ Predictions saved to predictions_all_models.csv")
print("✅ Metrics summary saved to model_metrics_summary.csv")
print("✅ All plots saved in 'plots/' folder")
print("✅ Consolidated PDF report saved as model_report.pdf")