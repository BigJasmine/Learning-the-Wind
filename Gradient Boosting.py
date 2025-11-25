from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)
print("GBR RMSE:", mean_squared_error(y_test, y_pred, squared=False))
print("GBR R²:", r2_score(y_test, y_pred))