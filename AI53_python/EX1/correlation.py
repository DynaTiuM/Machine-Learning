import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

file_path = os.path.join('EX1', 'data', 'ENB2012_data.xlsx')
df = pd.read_excel(file_path)

df_cleaned = df.dropna()

numeric_columns = [
    'X1',
    'X2',
    'X3',
    'X4',
    'X5',
    'X6',
    'X7',
    'X8',
    'Y1',
    'Y2'
]

correlation_matrix = df_cleaned[numeric_columns].corr()

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt='.2f', 
    linewidths=0.5
)
plt.title('Matrice de corrélation')
plt.show()

