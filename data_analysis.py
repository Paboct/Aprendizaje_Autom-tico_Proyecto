#Data analysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('train_students.csv')
print(df['Gender'].value_counts())
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
plt.xticks(rotation=0)

# Codificar las variables categóricas con Label Encoding para hacer la matriz de correlación
categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
le = LabelEncoder()
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Seleccionar solo las columnas numéricas para calcular la correlación
numerical_df = df.select_dtypes(include=['int64', 'float64'])

# Matriz de correlación
plt.figure(figsize=(15, 10))
sns.heatmap(numerical_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matriz de Correlación entre Variables Numéricas (con variables categóricas codificadas)")

# Countplot sobre la distancia de vuelo según la lealtad del cliente, agrupado por Clase
# Necesito reemplazar los valores numéricos por los valores originales. Para ello sustituyo por sus valores originales
df['Class'] = df['Class'].replace({0: 'Business', 1: 'Eco', 2: 'Eco Plus'})
df['Customer Type'] = df['Customer Type'].replace({0: 'Loyal Customer', 1: 'disloyal Customer'})

sns.catplot(x='Customer Type', y='Flight Distance', hue='Class', data=df, kind='bar')
plt.title("Distancia de Vuelo por Tipo de Cliente y Clase")
plt.xlabel("Tipo de Cliente")
plt.ylabel("Distancia de Vuelo")

# Distribución satisfacción según clase
df['satisfaction'] = df['satisfaction'].replace({0:'neutral or dissatisfied', 1:'satisfied'})

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='satisfaction', hue='Class')
plt.title("Satisfacción por Clase")
plt.xlabel("Satisfaction")
plt.ylabel("Total")#

#Histograma de la distancia de vuelo por satisfacción
plt.figure(figsize=(10,5))
sns.histplot(data=df, x='Flight Distance', kde=True, hue='satisfaction', alpha=0.4,  palette='viridis')
plt.title("Distribución de la Distancia de Vuelo por Satisfacción")
plt.xlabel("Distancia de Vuelo")
plt.ylabel("Total")
plt.axis([0,5000,0,3500])

#Hacer catplots respecto a la satisfacción para cada columna categórica
cols_object = df.select_dtypes(include=['object']).drop(columns='satisfaction').columns

plt.figure(figsize=(10,6))

for i, col in enumerate(cols_object):
    plt.subplot(1, 2, i + 1)  
    sns.countplot(data=df, x=col, hue='satisfaction', palette = "Set1")
    plt.title(f'Satisfacción respecto {col}')

plt.tight_layout()

# Agrupar datos por género y satisfacción
df['Gender'] = df['Gender'].replace({0:'Female', 1:'Male'})
satisfaction_gender = df.groupby(['Gender', 'satisfaction']).size()
print(satisfaction_gender)
# Dividir los datos por género
female_data = satisfaction_gender['Female']
male_data = satisfaction_gender['Male']

# Crear etiquetas y tamaños para cada género
female_labels = [f"{sat}" for sat in female_data.index]
female_sizes = female_data.values

male_labels = [f"{sat}" for sat in male_data.index]
male_sizes = male_data.values

# Crear paletas de colores para cada género
female_colors = sns.light_palette("red", n_colors=len(female_labels))
male_colors = sns.light_palette("blue", n_colors=len(male_labels))

# Crear la figura y los gráficos
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Gráfico para mujeres
#axes[0].pie(female_sizes, labels=female_labels, autopct='%1.1f%%', colors=female_colors)
axes[0].set_title("Satisfacción - Mujeres")

# Gráfico para hombres
axes[1].pie(male_sizes, labels=male_labels, autopct='%1.1f%%', colors=male_colors)
axes[1].set_title("Satisfacción - Hombres")

plt.tight_layout()

#Gráficas respecto a satisfacción
study_columns = df.iloc[:, 7:-3].columns

for col in study_columns:
    plt.figure(figsize=(10, 6))

    grouped_data = df.groupby(['satisfaction', col]).size().reset_index(name='count')
    
    sns.barplot(data=grouped_data, x=col, y='count', hue='satisfaction', palette='viridis')
    plt.title(f"Satisfacción respecto a {col}")
    plt.xlabel(col)
    plt.ylabel("Número de clientes")
    plt.legend(title="Nivel de satisfacción")
    
# Distribucion de los Retrasos



plt.show()
