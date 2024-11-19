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

## Hacer un plot sobre el tipo de clase y su satisfacción
df['satisfaction'] = df['satisfaction'].replace({0:'neutral or dissatisfied', 1:'satisfied'})

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='satisfaction', hue='Class')
plt.title("Satisfacción por Clase")
plt.xlabel("Satisfaction")
plt.ylabel("Total")

#Histograma de la distancia de vuelo por satisfacción
sns.histplot(data=df, x='Flight Distance', kde=True, hue='satisfaction', alpha=0.4,  palette='viridis')
plt.title("Distribución de la Distancia de Vuelo por Satisfacción")
plt.xlabel("Distancia de Vuelo")
plt.ylabel("Total")
plt.axis([0,5000,0,3500])

#Histograma satisfacción por limpieza
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Cleanliness', hue='satisfaction', palette='viridis')
plt.title('Customer Satisfaction by Cleanliness')
plt.xlabel('Cleanliness Rating')
plt.ylabel('Count')
plt.legend(title='Satisfaction Level')

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Ease of Online booking', hue='satisfaction', palette='viridis')
plt.title('Customer Satisfaction by Ease of Online Booking')
plt.xlabel('Ease of Online Booking Rating')
plt.ylabel('Number of Customers')
plt.legend(title='Satisfaction Level')

#Hacer catplots respecto a la satisfaccción para cada columna categórica
#Como incluir todas las columnas menos la de satisfacción
#cols = df.drop('satisfaction', axis=1).columns

cols = df.select_dtypes(include=['object']).columns
print(cols)

plt.figure(figsize=(12, 18))

for i, col in enumerate(cols):
    plt.subplot(1, 3, i + 1)  
    sns.countplot(data=df, x=col, hue='satisfaction', palette = "Set1")
    plt.title(f'Satisfacción respecto {col}')

plt.tight_layout()

plt.show()