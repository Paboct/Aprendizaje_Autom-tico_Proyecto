import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Diccionario para guardar las métricas de los modelos
ACCURACIES = {'Model': [], 'Accuracy': [], 'Train Score': [], 'Test Score': []}

# Carga del fichero csv de los datos
data = pd.read_csv('train_students.csv')

def transform_data_min_max(df: pd.DataFrame, labels_encoded: dict, scaler:MinMaxScaler) -> pd.DataFrame:
    """
    Transforma las columnas categóricas y numéricas de un DataFrame con un LabelEncoder y los
    escala mediante el escalado MinMax.
    """
    for col, le in labels_encoded.items():
        df[col] = le.transform(df[col]) if df[col].iloc[0] in le.classes_ else -1  # Asignar -1 si es desconocido

    numeric_columns = [col for col in df.columns if col in scaler.feature_names_in_]
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    return df

def transform_data_standard(df: pd.DataFrame, labels_encoded: dict, scaler:StandardScaler) -> pd.DataFrame:
    """
    Transforma las columnas categóricas y numéricas de un DataFrame con un LabelEncoder y 
    realiza el escalado standard.
    """
    for col, le in labels_encoded.items():
        df[col] = le.transform(df[col]) if df[col].iloc[0] in le.classes_ else -1  # Obtengo el valor de la columna y lo transformo

    numeric_columns = [col for col in df.columns if col in scaler.feature_names_in_]
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    return df

'''
#Creación de nuevas características
'''
#1. Total Delay
data['Total Delay'] = data['Departure Delay in Minutes'] + data['Arrival Delay in Minutes']
data.drop(columns=['Departure Delay in Minutes', 'Arrival Delay in Minutes'], inplace=True)

#2. Age en rangos
ranges = [0, 18, 35, 60, 100]
labels = ['Joven', 'Adulto Joven', 'Adulto', 'Adulto Mayor']

data['Age Groups'] = pd.cut(data['Age'], bins=ranges, labels=labels)
data.drop(columns='Age', inplace=True)

#3. Distancia en rangos
ranges = [0, 500, 1500, 3000, 5000, 10000]
labels = ['Short', 'Medium', 'Long', 'Very Long', 'Ultra Long']

data['Distance Range'] = pd.cut(data['Flight Distance'], bins=ranges, labels=labels)
data.drop(columns='Flight Distance', inplace=True)

#4. Comfort Total
data['Comfort Total'] = (data['Seat comfort'] + data['Inflight entertainment'] + data['Inflight service'] + data['Leg room service'] + data['On-board service'] + data['Cleanliness'] + data['Food and drink'] + data['Baggage handling'] + data['Checkin service'] + data['Inflight wifi service'] + data['Ease of Online booking'] + data['Departure/Arrival time convenient'] + data['Gate location'])/13
data.drop(columns= ['Seat comfort', 'Inflight entertainment', 'Inflight service', 'Leg room service', 'On-board service', 'Cleanliness', 'Food and drink', 'Baggage handling', 'Checkin service', 'Inflight wifi service', 'Ease of Online booking', 'Departure/Arrival time convenient', 'Gate location'], inplace=True)

df_preprocessed_standard = data.copy() #minmax
df_preprocessed_min_max = data.copy() #standard

'''
#LIMPIEZA DE DATOS
'''
#1 Manejo de valores nulos
#1.1 Rellenando con la media
#mean_delay = data['Arrival Delay in Minutes'].mean()
#df_preprocessed_standard = data.copy()
#df_preprocessed_standard['Arrival Delay in Minutes'].fillna(mean_delay, inplace=True)


#1.2 Para Total Delay
#df_preprocessed_standard = data.fillna(data['Total Delay'].mean())
#print(df_preprocessed_standard.isnull().sum())

#1.3 Eliminando los valores nulos
#df_preprocessed_standard = data.dropna()

#1.4 Usando KNNImputer
imputer = KNNImputer(n_neighbors=5)

#buscar que caracteristicas son nulas y rellenamos con el KNNimputer
for col in df_preprocessed_standard.columns:
    if df_preprocessed_standard[col].isnull().sum() > 0:
        df_preprocessed_standard[col] = imputer.fit_transform(df_preprocessed_standard[[col]])

#1.5 Eliminación de la columna no relevante
#df_preprocessed_standard.drop(columns='Departure Delay in Minutes', inplace=True)
#df_preprocessed_standard.drop(columns='id', inplace=True)

'''
# TRANSFORMACIÓN DE DATOS
'''
#Codificación de variables categóricas
category_columns = [col for col in df_preprocessed_standard.columns if df_preprocessed_standard[col].dtype == 'object' or df_preprocessed_standard[col].dtype == 'category']

for col in category_columns: #Ajustamos el LabelEncoder con todas las categorías posibles en los datos de entrenamiento
   le = LabelEncoder()
   df_preprocessed_standard[col] = le.fit_transform(df_preprocessed_standard[col])
   df_preprocessed_min_max[col] = le.fit_transform(df_preprocessed_min_max[col])

#Normalización de datos numéricos
numeric_columns = [col for col in df_preprocessed_standard.columns if df_preprocessed_standard[col].dtype != 'object' and col != 'satisfaction']

#Aplicar Min-Max Scaler
min_max_scaler = MinMaxScaler()
df_preprocessed_min_max[numeric_columns] = min_max_scaler.fit_transform(df_preprocessed_standard[numeric_columns])

#Apliación de Standard Scaler
standard_scaler = StandardScaler()
df_preprocessed_standard[numeric_columns] = standard_scaler.fit_transform(df_preprocessed_standard[numeric_columns])

'''
#SHOW ACCURACIES
'''
df_accuracies = pd.DataFrame(ACCURACIES)

# Pásamos los data frames a ficheros csv para ver el preprocesado
df_preprocessed_standard.to_csv('train_students_preprocessed_standard.csv', index=False)
df_preprocessed_min_max.to_csv('train_students_preprocessed_minmax.csv', index=False)