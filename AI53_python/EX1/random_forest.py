import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor

file_path = os.path.join('EX1', 'data', 'ENB2012_data.xlsx')
df = pd.read_excel(file_path)

df_cleaned = df.dropna()

X = df_cleaned.drop(columns=['Y1', 'Y2'])
y = df_cleaned['Y1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_model.fit(X_train_normalized, y_train)

y_pred_train_rf = rf_model.predict(X_train_normalized)
y_pred_test_rf = rf_model.predict(X_test_normalized)

train_mae_rf = mean_absolute_error(y_train, y_pred_train_rf)
test_mae_rf = mean_absolute_error(y_test, y_pred_test_rf)

print(f"Régression par forêt aléatoire - MAE sur l'ensemble d'entraînement : {train_mae_rf:.2f}")
print(f"Régression par forêt - MAE sur l'ensemble de test : {test_mae_rf:.2f}")


train_mse_rf = mean_squared_error(y_train, y_pred_train_rf)
test_mse_rf = mean_squared_error(y_test, y_pred_test_rf)

print(f"Régression par forêt aléatoire - MSE sur l'ensemble d'entraînement : {train_mse_rf:.2f}")
print(f"Régression par forêt - MSE sur l'ensemble de test : {test_mse_rf:.2f}")