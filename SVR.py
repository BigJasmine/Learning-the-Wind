import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("era5_features.csv", parse_dates=["time"])

# Features & target
X = df[["windspeed", "temperature_C", "pressure_hPa", "windspeed_roll24h", "windspeed_cubed"]].fillna(0)
y = df["windspeed"]  # Example target: windspeed itself or turbine output if available

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVR
svr = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train_scaled, y_train)

# Evaluate
y_pred = svr.predict(X_test_scaled)
print("SVR RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("SVR R²:", r2_score(y_test, y_pred))