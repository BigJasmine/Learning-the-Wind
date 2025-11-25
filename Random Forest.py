from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print("RF RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("RF R²:", r2_score(y_test, y_pred))