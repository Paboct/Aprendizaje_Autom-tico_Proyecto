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

ACCURACIES = {'Model': [], 'Accuracy': [], 'Train Score': [], 'Test Score': []}

# Carga el conjunto de datos
data = pd.read_csv('train_students.csv')

def transform_data_min_max(df: pd.DataFrame, labels_encoded: dict, scaler:MinMaxScaler) -> pd.DataFrame:
    """
    Transforma las columnas categóricas y numéricas de un DataFrame con un LabelEncoder y MinMaxScaler.
    """

    for col, le in labels_encoded.items():
        df[col] = le.transform(df[col]) if df[col].iloc[0] in le.classes_ else -1  # Asignar -1 si es desconocido

    numeric_columns = [col for col in df.columns if col in scaler.feature_names_in_]
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    return df

def transform_data_standard(df: pd.DataFrame, labels_encoded: dict, scaler:StandardScaler) -> pd.DataFrame:
    """
    Transforma las columnas categóricas y numéricas de un DataFrame con un LabelEncoder y StandardScaler.
    """
    for col, le in labels_encoded.items():
        df[col] = le.transform(df[col]) if df[col].iloc[0] in le.classes_ else -1  # Obtengo el valor de la columna y lo transformo

    numeric_columns = [col for col in df.columns if col in scaler.feature_names_in_]
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    return df

def LR_model(df:pd.DataFrame, model:str) -> LogisticRegression:
    """Realiza el modelo de Regresión Logística y
    nos devuelve el modelo"""
    X = df.drop(columns=['satisfaction'])
    y = df['satisfaction']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LR.classes_)
    disp.plot()
    plt.title(f"Confusion Matrix {model}")
    
    train_score = LR.score(X_train, y_train)
    test_score = LR.score(X_test, y_test)

    print(f"Train Accuracy Score: {train_score:.4f}")
    print(f"Test Accuracy Score: {test_score:.4f}")

    ACCURACIES['Model'].append(model)
    ACCURACIES['Accuracy'].append(accuracy)
    ACCURACIES['Train Score'].append(train_score)
    ACCURACIES['Test Score'].append(test_score)

    return LR

#Creación de nuevas características
#Total Delay
#data['Total Delay'] = data['Departure Delay in Minutes'] + data['Arrival Delay in Minutes']
#data.drop(columns=['Departure Delay in Minutes', 'Arrival Delay in Minutes'], inplace=True)

#Age en rangos
ranges = [0, 18, 35, 60, 100]
labels = ['Joven', 'Adulto Joven', 'Adulto', 'Adulto Mayor']

data['Age Groups'] = pd.cut(data['Age'], bins=ranges, labels=labels)
data.drop(columns='Age', inplace=True)

#print(data.shape)
#print(data['Age Groups'].value_counts().sum)
#print(data.head(30))

#Distancia en rangos
ranges = [0, 500, 1500, 3000, 5000, 10000]
labels = ['Short', 'Medium', 'Long', 'Very Long', 'Ultra Long']

data['Distance Range'] = pd.cut(data['Flight Distance'], bins=ranges, labels=labels)
data.drop(columns='Flight Distance', inplace=True)

#Comfort Total
data['Comfort Total'] = data['Seat comfort'] + data['Inflight entertainment'] + data['Inflight service'] + data['Leg room service']
data.drop(columns=['Seat comfort', 'Inflight entertainment', 'Inflight service', 'Leg room service'], inplace=True)

#LIMPIEZA DE DATOS
# Manejo de valores nulos
#Rellenando con la media
mean_delay = data['Arrival Delay in Minutes'].mean()
df_preprocessed_1 = data.copy()
df_preprocessed_1['Arrival Delay in Minutes'].fillna(mean_delay, inplace=True)


#Para Total Delay
#df_preprocessed_1 = data.fillna(data['Total Delay'].mean())
#print(df_preprocessed_1.isnull().sum())

#Eliminando los valores nulos
#df_preprocessed_1 = data.dropna()

#Usando KNNImputer
imputer = KNNImputer(n_neighbors=5)
df_preprocessed_1 = data.copy()
df_preprocessed_1['Departure Delay in Minutes'] = imputer.fit_transform(df_preprocessed_1[['Departure Delay in Minutes']])
df_preprocessed_1['Arrival Delay in Minutes'] = imputer.fit_transform(df_preprocessed_1[['Arrival Delay in Minutes']])

#Eliminación de la columna no relevante
#df_preprocessed_1.drop(columns='Departure Delay in Minutes', inplace=True)
#df_preprocessed_1.drop(columns='id', inplace=True)

print(df_preprocessed_1.info())
df_preprocessed_2 = df_preprocessed_1.copy()

# Comprobación de valores nulos
print("Missing Values:")
print(df_preprocessed_1.isnull().sum(), "\n")

# Comprobación de forma del DataFrame
print(f"Original Shape: {data.shape}")
print(f"Shape after handling missing values: {df_preprocessed_1.shape}\n")

# Tipos de datos
print("Data Types:")
print(df_preprocessed_1.dtypes, "\n")

# TRANSFORMACIÓN DE DATOS
# Codificación de variables categóricas
category_columns = [col for col in df_preprocessed_1.columns if df_preprocessed_1[col].dtype == 'object' or df_preprocessed_1[col].dtype == 'category']

## Inicializamos un diccionario para guardar los LabelEncoders de cada columna
labels_encoded = {}

# Ajustamos el LabelEncoder con todas las categorías posibles en los datos de entrenamiento
for col in category_columns:
   le = LabelEncoder()
   df_preprocessed_1[col] = le.fit_transform(df_preprocessed_1[col])
   df_preprocessed_2[col] = le.fit_transform(df_preprocessed_2[col])
   labels_encoded[col] = le

##Eliminamos la columna satisfaction del diccionario, ya que las predicciones no se harán sobre esta columna
labels_encoded.pop('satisfaction', None)

print("Categorical Variables Encoded:")
print(df_preprocessed_1.head(), "\n")

#Normalización de datos numéricos
numeric_columns = [col for col in df_preprocessed_1.columns if df_preprocessed_1[col].dtype != 'object' and col != 'satisfaction']

# Aplicar Min-Max Scaler
min_max_scaler = MinMaxScaler()
df_preprocessed_2[numeric_columns] = min_max_scaler.fit_transform(df_preprocessed_1[numeric_columns])

#Apliación de Standard Scaler
standard_scaler = StandardScaler()
df_preprocessed_1[numeric_columns] = standard_scaler.fit_transform(df_preprocessed_1[numeric_columns])

print("Data Normalized:")
# Comprobación de los datos normalizados
#print(f"Mínimos y Máximos de las columnas normalizadas:\n{df_preprocessed_1[numeric_columns].min()}\n{df_preprocessed_1[numeric_columns].max()}\n")
print(f"Standard deviation: \n{df_preprocessed_1.std()}\n")
print(f"Information of the normalized dataset:\n{df_preprocessed_1.info()}\n")

# Distribución de la variable objetivo
#print("Distribution of Target Variable:")
#print(df_preprocessed_1['satisfaction'].value_counts(normalize=True), "\n")

#Realizmaos el modelo de Regresión Logística con los datos preprocesados
LR_stesc = LR_model(df_preprocessed_1, 'Standard Scaler')
LR_min = LR_model(df_preprocessed_2, 'Min-Max Scaler')
plt.show()

#SHOW ACCURACIES
df_accuracies = pd.DataFrame(ACCURACIES)
print(df_accuracies,"\nAlgunas Predicciones:")

# Pásamos los data frames a ficheros csv
df_preprocessed_1.to_csv('train_students_preprocessed_standard.csv', index=False)
df_preprocessed_2.to_csv('train_students_preprocessed_minmax.csv', index=False)

#example = pd.DataFrame({
#   'id': [12345],
#   'Gender': ['Female'],
#   'Customer Type': ['Loyal Customer'],
#   'Age': [35],
#   'Type of Travel': ['Business travel'],
#   'Class': ['Business'],
#   'Flight Distance': [1200],
#   'Inflight wifi service': [4],
#   'Departure/Arrival time convenient': [4],
#   'Ease of Online booking': [5],
#   'Gate location': [3],
#   'Food and drink': [4],
#   'Online boarding': [5],
#   'Seat comfort': [3],
#   'Inflight entertainment': [2],
#   'On-board service': [4],
#   'Leg room service': [5],
#   'Baggage handling': [4],
#   'Checkin service': [4],
#   'Inflight service': [3],
#   'Cleanliness': [4],
#   'Departure Delay in Minutes': [10],
#   'Arrival Delay in Minutes': [5],
#})
#
#example = transform_data_min_max(example, labels_encoded, min_max_scaler)
#example = transform_data_standard(example, labels_encoded, standard_scaler)
#prediction_sts = LR_stesc.predict(example)
#prediction_minmax = LR_min.predict(example)
#print("Example 1")
#print(f"Predicted Satisfaction with Standard Scaler: {'neutral or insatisfied' if prediction_sts[0] == 0 else 'satisfied'}")
#print(f"Predicted Satisfaction with Min-Max Scaler: {'neutral or insatisfied' if prediction_minmax[0] == 0 else 'satisfied'}")
#
#example_2 = pd.DataFrame({
#   'id': [67890],
#   'Gender': ['Male'],
#   'Customer Type': ['Loyal Customer'],
#   'Age': [42],
#   'Type of Travel': ['Personal Travel'],
#   'Class': ['Economy'],
#   'Flight Distance': [850],
#   'Inflight wifi service': [3],
#   'Departure/Arrival time convenient': [5],
#   'Ease of Online booking': [4],
#   'Gate location': [2],
#   'Food and drink': [3],
#   'Online boarding': [4],
#   'Seat comfort': [2],
#   'Inflight entertainment': [3],
#   'On-board service': [3],
#   'Leg room service': [4],
#   'Baggage handling': [3],
#   'Checkin service': [5],
#   'Inflight service': [3],
#   'Cleanliness': [4],
#   'Departure Delay in Minutes': [0],
#   'Arrival Delay in Minutes': [8],
#})
#
#example_2 = transform_data_min_max(example_2, labels_encoded, min_max_scaler)
#example_2 = transform_data_standard(example_2, labels_encoded, standard_scaler)
#prediction_sts = LR_stesc.predict(example_2)
#prediction_minmax = LR_min.predict(example_2)
#print("Example 2")
#print(f"Predicted Satisfaction with Standard Scaler: {'neutral or insatisfied' if prediction_sts[0] == 0 else 'satisfied'}")
#print(f"Predicted Satisfaction with Min-Max Scaler: {'neutral or insatisfied' if prediction_minmax[0] == 0 else 'satisfied'}")


#example_3 = pd.DataFrame({
#   'id': [67890],
#   'Gender': ['Female'],
#   'Customer Type': ['Loyal Customer'],
#   'Age': [35],
#   'Type of Travel': ['Business travel'],
#   'Class': ['Business'],
#   'Flight Distance': [1200],
#   'Inflight wifi service': [4],
#   'Departure/Arrival time convenient': [4],
#   'Ease of Online booking': [5],
#   'Gate location': [4],
#   'Food and drink': [5],
#   'Online boarding': [5],
#   'Seat comfort': [4],
#   'Inflight entertainment': [5],
#   'On-board service': [5],
#   'Leg room service': [4],
#   'Baggage handling': [5],
#   'Checkin service': [5],
#   'Inflight service': [5],
#   'Cleanliness': [5],
#   'Departure Delay in Minutes': [0],
#   'Arrival Delay in Minutes': [5],
#})
#
#example_3 = transform_data_min_max(example_3, labels_encoded, min_max_scaler)
#example_3 = transform_data_standard(example_3, labels_encoded, standard_scaler)
#prediction_sts = LR_stesc.predict(example_3)
#prediction_minmax = LR_min.predict(example_3)
#print("Example 3")
#print(f"Predicted Satisfaction with Standard Scaler: {'neutral or insatisfied' if prediction_sts[0] == 0 else 'satisfied'}")
#print(f"Predicted Satisfaction with Min-Max Scaler: {'neutral or insatisfied' if prediction_minmax[0] == 0 else 'satisfied'}")