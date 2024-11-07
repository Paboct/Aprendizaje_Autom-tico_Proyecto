import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#REGRESION LINEAL
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


df = pd.read_csv("train_new.csv")
# Imprimir los valores nulos por columna antes de la limpieza 1
valores_nulos_por_columna = df.isnull().sum(axis=0)
print("Valores nulos por columna:")
print(valores_nulos_por_columna)
suma_total_nulos = df.isnull().sum().sum()
#imprimir los valores nulos totales
print('Total: ', suma_total_nulos)
#ELIMINAMOS LAS FILAS QUE LES FALTEN UN 80% DE LOS DATOS (QUE TENGAN 11 VALORES VACIOS)
df_filtrado = df.dropna(thresh=3)
#TRAS UNA INSPECCION DEL DF ORIGINAL NOS DAMOS CUENTA DE QUE 
#1.SI CRYOSLEEP == TRUE SEGURAMENTE VIP==FALSE
#2.SI CRYOSLEEP == TRUE & VIP == FALSE SEGURAMENTE LOS GASTOS ECONOMICOS == 0
#3.SI LOS GASTOS ECONOMICOS == 0 SEGURAMENTE CRYOSLEEP == TRUE
#4.SI CRYOSLEEP== TRUE & VIP==TRUE SEGURAMENTE HOMEPLANET==EUROPA ¿COINCIDENCIA?

#1. RELLENAMOS ASI 73 VALORES
condicion = (df['CryoSleep'] == True)
df.loc[condicion, 'VIP'] = df.loc[condicion, 'VIP'].fillna(False)

#2. RELLENAMOS ASI 336 VALORES
condicion = (df['CryoSleep'] == True) & (df['VIP'] == False) & (df[['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].isnull().any(axis=1))
df.loc[condicion, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = df.loc[condicion, ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']].fillna(0)

#3. RELLENAMOS ASI 224 VALORES
condicion = (df['RoomService'] == 0) & (df['FoodCourt'] == 0) & (df['ShoppingMall'] == 0) & (df['Spa'] == 0) & (df['VRDeck'] == 0)
df.loc[condicion, 'CryoSleep'] = df.loc[condicion, 'CryoSleep'].fillna(True)

#4
condicion = (df['CryoSleep'] == True) & (df['VIP'] == True)
df.loc[condicion, 'HomePlanet'] = df.loc[condicion, 'HomePlanet'].fillna('Europa')

# Imprimir los valores nulos por columna antes de la limpieza 1
valores_nulos_por_columna = df.isnull().sum(axis=0)
print("Valores nulos por columna:")
print(valores_nulos_por_columna)
#imprimir los valores nulos totales
suma_total_nulos = df.isnull().sum().sum()
print('Total: ', suma_total_nulos)

#COMO NO HE ENCONTRADO MAS RELACIONES A PRIMERA VISTA VOY A RELLENAR LOS VALORES NULOS DE LA SIGUIENTE FORMA:
#HOMEPLANTET --> MODA (ALEATORIO)
#CRYOSLEEP --> MODA (ALEATORIO)
#CABIN --> ¿QUITARLOS?¿MODA? son solo 183
#DESTINATION --> MODA (se puede probar con aleatorio)
#AGE --> ALEATORIO (se puede probar con media)
#VIP --> MODA
#GASTOS ECONOMICOS --> PONER A 0 (MEDIA, MODA)
#NAME --> de momento no lo vamos a utilizar , pq no he probado lo de las familias de momento
#ID --> same que lo de names

#ETIQUETA --> regresion lineal????

"""
#division de grupos, para ver si aproximando por familias va mejor que haciendo simplemente moda
list_grupo = []
list_num_grupo = []

for ID in df['PassengerId']:
    grupo, numero_en_grupo = ID.split('_')
    list_grupo.append(grupo)
    list_num_grupo.append(numero_en_grupo)

#comprobar cuantos grupos diferentes hay
numero_grupos = len(set(list_grupo))
print("el numero de grupos que hay es:", numero_grupos)
"""





#HOMEPLANET --> MODA
# Calcular la moda de la columna 'HomePlanet'
moda_homeplanet = df['HomePlanet'].mode()[0] #df[] -> linea de la moda, [0] -> si hay varios valores que se repiten lo mismo devuelve el primero
# Rellenar los valores nulos con la moda
df['HomePlanet'] = df['HomePlanet'].fillna(moda_homeplanet)

#CRYOSLEEP --> MODA
# Calcular la moda de la columna 'CryoSleep'
moda_cryo = df['CryoSleep'].mode()[0]
# Rellenar los valores nulos con la moda
df['CryoSleep'] = df['CryoSleep'].fillna(moda_cryo)


#CABIN --> MODA (AUNQUE PLANTEARSE QUITAR LAS FILAS CON VALORES NULOS)
# Calcular la moda de la columna 'Cabin'
moda_cabin = df['Cabin'].mode()[0]
# Rellenar los valores nulos con la moda
df['Cabin'] = df['Cabin'].fillna(moda_cabin)

#df = df.dropna(subset=['Cabin']) #QUITAR LAS FILAS CON CABIN NULO

#DESTINATION --> MODA 
# Calcular la moda de la columna 'Destination'
moda_desty = df['Destination'].mode()[0]
# Rellenar los valores nulos con la moda
df['Destination'] = df['Destination'].fillna(moda_desty)

#AGE --> RANDOM
# Obtener la cantidad de valores nulos en la columna 'Age'
cantidad_nulos = df['Age'].isnull().sum()

# Generar números aleatorios en el rango de edad deseado
rango_edad = (df['Age'].mean() - df['Age'].std(), df['Age'].mean() + df['Age'].std()) #genera números aleatorios en un rango definido por la media y la desviación estándar de la columna 'Age'. 
valores_aleatorios = np.random.uniform(rango_edad[0], rango_edad[1], cantidad_nulos)

# Rellenar los valores nulos con los números aleatorios
df.loc[df['Age'].isnull(), 'Age'] = valores_aleatorios

#VIP --> MODA
# Calcular la moda de la columna 'VIP'
moda_vip = df['VIP'].mode()[0]
# Rellenar los valores nulos con la moda
df['VIP'] = df['VIP'].fillna(moda_vip)


#GASTOS ECONOMICOS --> PONERLAS A 0
# Lista de columnas a procesar
columnas_a_rellenar_con_cero = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
# Rellenar los valores nulos con 0 en las columnas especificadas
df[columnas_a_rellenar_con_cero] = df[columnas_a_rellenar_con_cero].fillna(0)

#NOMBRE E ID --> NO LOS VAMOS A UTILIZAR TODAVIA ASIQ LOS ELIMINAMOS
df = df.drop(['PassengerId', 'Name'], axis=1)

#ETIQUETAS --> REGRESION
#antes de hacer regresion vamos a pasar a valores numericos las caractaeristicas
columnas_a_codificar = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"]

# Crear una instancia de LabelEncoder
labelencoder = LabelEncoder()

# Aplicar LabelEncoder solo a las columnas seleccionadas
for columna in columnas_a_codificar:
    df[columna] = labelencoder.fit_transform(df[columna])


# Dividir el DataFrame en dos conjuntos
df_con_etiquetas = df.dropna(subset=['Transported'])  # Conjunto con etiquetas existentes
df_sin_etiquetas = df[df['Transported'].isnull()]     # Conjunto con valores nulos en las etiquetas

# Dividir los conjuntos en características (X) y etiquetas (y)
X_train = df_con_etiquetas.drop(['Transported'], axis=1)
y_train = df_con_etiquetas['Transported']
X_test = df_sin_etiquetas.drop(['Transported'], axis=1)

# Entrenar un modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predecir las etiquetas faltantes
predicciones = modelo.predict(X_test)

# Asignar las predicciones al DataFrame original
# Aplicar umbral para convertir las predicciones a True o False
umbral = 0.5
df.loc[df['Transported'].isnull(), 'Transported'] = predicciones > umbral



# Imprimir los valores nulos por columna antes de la limpieza 2
valores_nulos_por_columna = df.isnull().sum(axis=0)
print("Valores nulos por columna:")
print(valores_nulos_por_columna)
#imprimir los valores nulos totales
suma_total_nulos = df.isnull().sum().sum()
print('Total: ', suma_total_nulos)


#GUARDAMOS EL DF LIMPIO EN UN NUEVO EXCEL 
nombre_archivo_csv = 'excel_limpio.csv'
df.to_csv(nombre_archivo_csv, index=False)
