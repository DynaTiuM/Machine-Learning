import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

file_path = os.path.join('EX1', 'data', 'ENB2012_data.xlsx')
df = pd.read_excel(file_path)

df_cleaned = df.dropna()

X = df_cleaned.drop(columns=['Y1', 'Y2'])
y = df_cleaned['Y1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

linear_model = LinearRegression()
linear_model.fit(X_train_normalized, y_train)

y_pred_train = linear_model.predict(X_train_normalized)
y_pred_test = linear_model.predict(X_test_normalized)

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)

print(f"Régression linéaire - MSE sur l'ensemble d'entraînement : {train_mse:.2f}")
print(f"Régression linéaire - MSE sur l'ensemble de test : {test_mse:.2f}")

train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"Régression linéaire - MAE sur l'ensemble d'entraînement : {train_mae:.2f}")
print(f"Régression linéaire - MAE sur l'ensemble de test : {test_mae:.2f}")
