import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

#def transform_data_min_max(df: pd.DataFrame, labels_encoded: dict, scaler:MinMaxScaler) -> pd.DataFrame:
#    """
#    Transforma las columnas categóricas y numéricas de un DataFrame con un LabelEncoder y MinMaxScaler.
#    """
#
#    for col, le in labels_encoded.items():
#        df[col] = le.transform(df[col]) if df[col].iloc[0] in le.classes_ else -1  # Asignar -1 si es desconocido
#
#    numeric_columns = [col for col in df.columns if col in scaler.feature_names_in_]
#    df[numeric_columns] = scaler.transform(df[numeric_columns])
#
#    return df

def transform_data_standard(df: pd.DataFrame, labels_encoded: dict, scaler:StandardScaler) -> pd.DataFrame:
    """
    Transforma las columnas categóricas y numéricas de un DataFrame con un LabelEncoder y StandardScaler.
    """
    for col, le in labels_encoded.items():
        df[col] = le.transform(df[col]) if df[col].iloc[0] in le.classes_ else -1  # Asignar -1 si es desconocido

    numeric_columns = [col for col in df.columns if col in scaler.feature_names_in_]
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    return df

# Carga el conjunto de datos
data = pd.read_csv('train_students.csv')

# Manejo de valores nulos
#Rellenando con la media
mean_delay = data['Arrival Delay in Minutes'].mean()
df_preprocessed = data.fillna(mean_delay)
#Eliminando los valores nulos
#df_preprocessed = data.dropna()
#Usando KNNImputer
#imputer = KNNImputer(n_neighbors=5)
#df_preprocessed = data.copy()
#df_preprocessed['Departure Delay in Minutes'] = imputer.fit_transform(df_preprocessed[['Departure Delay in Minutes']])
#df_preprocessed['Arrival Delay in Minutes'] = imputer.fit_transform(df_preprocessed[['Arrival Delay in Minutes']])

# Comprobación de valores nulos
print("Missing Values:")
print(df_preprocessed.isnull().sum(), "\n")

# Comprobación de forma del DataFrame
print(f"Original Shape: {data.shape}")
print(f"Shape after handling missing values: {df_preprocessed.shape}\n")

# Tipos de datos
print("Data Types:")
print(df_preprocessed.dtypes, "\n")

# Codificación de variables categóricas
category_columns = [col for col in df_preprocessed.columns if df_preprocessed[col].dtype == 'object']

# Inicializamos un diccionario para guardar los LabelEncoders de cada columna
labels_encoded = {}

# Ajustamos el LabelEncoder con todas las categorías posibles en los datos de entrenamiento
for col in category_columns:
    le = LabelEncoder()
    df_preprocessed[col] = le.fit_transform(df_preprocessed[col])
    labels_encoded[col] = le

#Eliminamos la columna satisfaction del diccionario, ya que las predicciones no se harán sobre esta columna
labels_encoded.pop('satisfaction', None)

print("Categorical Variables Encoded:")
print(df_preprocessed.head(), "\n")

# Normalización de datos numéricos
numeric_columns = [col for col in df_preprocessed.columns if df_preprocessed[col].dtype != 'object' and col != 'satisfaction']

# Aplicar Min-Max Scaler
#min_max_scaler = MinMaxScaler()
#df_preprocessed[numeric_columns] = min_max_scaler.fit_transform(df_preprocessed[numeric_columns])

#Apliación de Standard Scaler
standard_scaler = StandardScaler()
df_preprocessed[numeric_columns] = standard_scaler.fit_transform(df_preprocessed[numeric_columns])

print("Data Normalized:")
# Comprobación de los datos normalizados
#print(f"Mínimos y Máximos de las columnas normalizadas:\n{df_preprocessed[numeric_columns].min()}\n{df_preprocessed[numeric_columns].max()}\n")
print(f"Standard deviation: \n{df_preprocessed.std()}\n")
print(f"Information of the normalized dataset:\n{df_preprocessed.info()}\n")

# Distribución de la variable objetivo
print("Distribution of Target Variable:")
print(df_preprocessed['satisfaction'].value_counts(normalize=True), "\n")

#Eliminación de la columna no relevante
#df_preprocessed.drop(columns='Departure Delay in Minutes', inplace=True)
#df_preprocessed.drop(columns='id', inplace=True)

# Separación de características (X) y objetivo (y)
X = df_preprocessed.drop(columns=['satisfaction'])
y = df_preprocessed['satisfaction']

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelo de Regresión Logística
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LR.classes_)
disp.plot()
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()

print(f"Train Accuracy Score: {LR.score(X_train, y_train):.4f}")
print(f"Test Accuracy Score: {LR.score(X_test, y_test):.4f}")

# Ejemplos de predicción

example = pd.DataFrame({
    'id': [12345],
    'Gender': ['Female'],
    'Customer Type': ['Loyal Customer'],
    'Age': [35],
    'Type of Travel': ['Business travel'],
    'Class': ['Business'],
    'Flight Distance': [1200],
    'Inflight wifi service': [4],
    'Departure/Arrival time convenient': [4],
    'Ease of Online booking': [5],
    'Gate location': [3],
    'Food and drink': [4],
    'Online boarding': [5],
    'Seat comfort': [3],
    'Inflight entertainment': [2],
    'On-board service': [4],
    'Leg room service': [5],
    'Baggage handling': [4],
    'Checkin service': [4],
    'Inflight service': [3],
    'Cleanliness': [4],
    'Departure Delay in Minutes': [10],
    'Arrival Delay in Minutes': [5],
})

#example = transform_data_min_max(example, labels_encoded, min_max_scaler)
example = transform_data_standard(example, labels_encoded, standard_scaler)
prediction = LR.predict(example)
print(f"Predicted Satisfaction: {'neutral or insatisfied' if prediction[0] == 0 else 'satisfied'}")

example_2 = pd.DataFrame({
    'id': [67890],
    'Gender': ['Male'],
    'Customer Type': ['Loyal Customer'],
    'Age': [42],
    'Type of Travel': ['Personal Travel'],
    'Class': ['Economy'],
    'Flight Distance': [850],
    'Inflight wifi service': [3],
    'Departure/Arrival time convenient': [5],
    'Ease of Online booking': [4],
    'Gate location': [2],
    'Food and drink': [3],
    'Online boarding': [4],
    'Seat comfort': [2],
    'Inflight entertainment': [3],
    'On-board service': [3],
    'Leg room service': [4],
    'Baggage handling': [3],
    'Checkin service': [5],
    'Inflight service': [3],
    'Cleanliness': [4],
    'Departure Delay in Minutes': [0],
    'Arrival Delay in Minutes': [8],
})


#example_2 = transform_data_min_max(example_2, labels_encoded, min_max_scaler)
example_2 = transform_data_standard(example_2, labels_encoded, standard_scaler)
prediction_2 = LR.predict(example_2)
print(f"Predicted Satisfaction for Example 2: {'neutral or insatisfied' if prediction_2[0] == 0 else 'satisfied'}")

example_3 = pd.DataFrame({
    'id': [67890],
    'Gender': ['Female'],
    'Customer Type': ['Loyal Customer'],
    'Age': [35],
    'Type of Travel': ['Business travel'],
    'Class': ['Business'],
    'Flight Distance': [1200],
    'Inflight wifi service': [4],
    'Departure/Arrival time convenient': [4],
    'Ease of Online booking': [5],
    'Gate location': [4],
    'Food and drink': [5],
    'Online boarding': [5],
    'Seat comfort': [4],
    'Inflight entertainment': [5],
    'On-board service': [5],
    'Leg room service': [4],
    'Baggage handling': [5],
    'Checkin service': [5],
    'Inflight service': [5],
    'Cleanliness': [5],
    'Departure Delay in Minutes': [0],
    'Arrival Delay in Minutes': [5],
})

#example_3 = transform_data_min_max(example_3, labels_encoded, min_max_scaler)
example_3 = transform_data_standard(example_3, labels_encoded, standard_scaler)
prediction_3 = LR.predict(example_3)
print(f"Predicted Satisfaction for Example 3: {'neutral or insatisfied' if prediction_3[0] == 0 else 'satisfied'}")