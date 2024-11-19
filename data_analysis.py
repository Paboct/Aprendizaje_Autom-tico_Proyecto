#Data analysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train_students.csv')
print(f"First data of the DataFrame:\n{df.head()}\n")
print(df['Class'].value_counts())
#Information of the dataset
print(f"Info:\n{df.info}\n")

#Información sobre la estadística
print(f"Stadistical data:\n{df.describe()}\n")

# Tamaño del dataset
print(f"Number of data: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")

# Valores nulos
print(f"Number of null values per feature:\n{df.isnull().sum()}\n")

# Distribución de valores en columnas categóricas
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in categorical_columns:
    print(f"Distribución de valores en {col}:\n{df[col].value_counts()}\n")

# Distribución de edad
plt.figure(figsize=(10, 5))
df['Age'].plot(kind='hist', bins=20, color='blue', alpha=0.7, width=3, align='mid')
plt.title("Distribución de Edad")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.xticks(np.arange(0, 90, 5))

# Distribución de la distancia del vuelo
plt.figure(figsize=(10, 5))
df['Flight Distance'].plot(kind='hist', bins=20, color='blue', alpha=0.7, width=100, align='mid')
plt.title("Distribución de la Distancia del Vuelo")
plt.xlabel("Distancia")
plt.ylabel("Frecuencia")
plt.axis([0, 5000, 0, 20000])
plt.xticks(np.arange(0, 5000, 250))
plt.yticks(np.arange(0, 20000, 1000))

# Distribución de la satisfacción
plt.figure(figsize=(6, 4))
df['satisfaction'].value_counts().plot(kind='bar', color='blue', alpha=0.7, align='center')
plt.title("Distribución de la Satisfacción del Pasajero")
plt.xlabel("Satisfacción")
plt.ylabel("Número de Pasajeros")

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

# Catplot de la distancia de vuelo por tipo de cliente y clase
# Necesito reemplazar los valores numéricos por los valores originales
df['Class'] = df['Class'].replace({0: 'Business', 1: 'Eco', 2: 'Eco Plus'})
df['Customer Type'] = df['Customer Type'].replace({0: 'Loyal Customer', 1: 'disloyal Customer'})

sns.catplot(x='Customer Type', y='Flight Distance', hue='Class', data=df, kind='bar')
plt.title("Distancia de Vuelo por Tipo de Cliente y Clase")
plt.xlabel("Tipo de Cliente")
plt.ylabel("Distancia de Vuelo")

# Hacer un plot sobre la satisfacción por edad y tipo de viaje
#plt.figure(figsize=(10, 5))
sns.catplot(x='satisfaction', y='Age', col='Type of Travel', data=df, kind='box')
plt.suptitle("Satisfacción por Edad y Tipo de Viaje")

## Hacer un plot sobre el tipo de clase y su satisfacción
df['satisfaction'] = df['satisfaction'].replace({0:'neutral or dissatisfied', 1:'satisfied'})

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='satisfaction', hue='Class')
plt.title("Satisfacción por Clase")
plt.xlabel("Satisfaction")
plt.ylabel("Total")
plt.show()