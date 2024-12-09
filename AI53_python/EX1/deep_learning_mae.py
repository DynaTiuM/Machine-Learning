import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

file_path = os.path.join('EX1', 'data', 'ENB2012_data.xlsx')
df = pd.read_excel(file_path)

df_cleaned = df.dropna()

X = df_cleaned.drop(columns=['Y1', 'Y2'])
y = df_cleaned['Y1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=X_train_normalized.shape[1], kernel_regularizer=l2(0.005)))
model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.003)))

model.add(Dense(units=1, activation="linear"))

model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')


history = model.fit(X_train_normalized, y_train, epochs=100, batch_size=64, validation_data=(X_test_normalized, y_test), verbose=1)

y_pred_train_nn = model.predict(X_train_normalized)
y_pred_test_nn = model.predict(X_test_normalized)

train_mae_nn = mean_absolute_error(y_train, y_pred_train_nn)
test_mae_nn = mean_absolute_error(y_test, y_pred_test_nn)

print(f"Réseau de neurones - MAE sur l'ensemble d'entraînement : {train_mae_nn:.2f}")
print(f"Réseau de neurones - MAE sur l'ensemble de test : {test_mae_nn:.2f}")


new_data = pd.DataFrame([[0.90, 563.50, 318.50, 122.50, 7.00, 4, 0.00, 0]], 
                        columns=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])

new_data_normalized = scaler.transform(new_data)

prediction = model.predict(new_data_normalized)

print(f"Prédiction pour l'exemple : {prediction[0][0]:.2f}")


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r--', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()