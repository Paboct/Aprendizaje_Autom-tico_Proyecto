#Data analysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train_students.csv')
print(f"Data:\n{df.head()}\n")
print(f"Info:\n{df.info()}\n")
print(f"Stadistical data:\n{df.describe()}\n")

# Plotting
# Tamaño del dataset
print("Número de filas y columnas:", df.shape)

# Primeras filas del dataset
print("Primeras filas del dataset:")
print(df.head())

# Información general del dataset
print("Información del dataset:")
print(df.info())

# Valores nulos
print("Valores nulos en cada columna:")
print(df.isnull().sum())

# Estadísticas descriptivas
print("Estadísticas descriptivas:")
print(df.describe())

# Distribución de valores en columnas categóricas
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in categorical_columns:
    print(f"Distribución de valores en {col}:")
    print(df[col].value_counts())
    print()

# Distribución de edad
plt.figure(figsize=(10, 5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Distribución de Edad")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

# Distribución de la distancia del vuelo
plt.figure(figsize=(10, 5))
sns.histplot(df['Flight Distance'], bins=20, kde=True)
plt.title("Distribución de la Distancia del Vuelo")
plt.xlabel("Distancia")
plt.ylabel("Frecuencia")
plt.show()

# Distribución de la satisfacción
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='satisfaction')
plt.title("Distribución de la Satisfacción del Pasajero")
plt.xlabel("Satisfacción")
plt.ylabel("Número de Pasajeros")
plt.show()

# Codificar las variables categóricas con Label Encoding
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Seleccionar solo las columnas numéricas para calcular la correlación
numerical_df = df.select_dtypes(include=['int64', 'float64'])

# Calcular la matriz de correlación
plt.figure(figsize=(15, 10))
sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación entre Variables Numéricas (con variables categóricas codificadas)")
plt.show()