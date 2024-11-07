import pandas as pd
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


data = pd.read_csv("train_new.csv")
df_con_na = pd.DataFrame(data)
df = df_con_na.dropna()

#1 cuantos datos y caracteristicas tiene
num_datos, num_caracteristicas = data.shape
print("el numero de datos es: ", num_datos)
print("\nel numero de caracteristicas es: ", num_caracteristicas)

#2 mostrar cada tipo de dato
tipo_dato = data.dtypes
print("\nel tipo de los datos es:\n",tipo_dato)

#3 mostrar estadisticas 
estadisticas = data.describe()
print("\nlas estadisticas son:",estadisticas)

"""
#5 normalizar los datos en el rango (0,1)
X = data.values[:,0:13]
y = data.values[:,13]
"""

#usamos oneHotEncoder para poder normalizar
#columnas a modificar
columnas_categoricas = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'Name']

#crear label encoder
le = LabelEncoder()

for columna in columnas_categoricas:
    df[columna] = le.fit_transform(df[columna])
    print(df[columna])




# Normalizar las características numéricas
numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("fadsfaf")
print(df.shape)


#divides en etiqueta y no etiqueta
X = df.values[:,0:13]
y = df.values[:,13]

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


print("iniciar entrenamiento")
# Inicializar y entrenar el modelo de regresión logística
model = LogisticRegression() #aqui me da el error de los datos faltantes
model.fit(X_train, y_train)

print("predicciones")
# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy}')