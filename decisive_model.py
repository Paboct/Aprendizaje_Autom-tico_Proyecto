import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score ,recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from characteristics_selection import select_kbest
from sklearn.impute import KNNImputer

def create_model(data:pd.DataFrame, n_folds:int=5) -> None:
    """Esta función crea un modelo neuronal y lo entrena"""
    #Dataframe del modelo
    df_neuron = {}

    #Listas de los accuracies de cada modelo
    neuron_accuracy = []
    neuron_f1 = []
    neuron_precision = []
    neuron_recall = []
    neuron_error = []
    neuron_train, neuron_test = [], []

    #Creamos los folds para realizar los entrenamientos
    k_fold_list = kfold(data, n_folds)

    """Neuronal Network"""
    #Creamos las listas de las métricas de cada fold
    neuron_fold_accuracy = []
    neuron_fold_f1 = []
    neuron_fold_precision = []
    neuron_fold_recall = []
    neuron_fold_error = []

    for train, test in k_fold_list:
        
        x_train = train.drop(columns=["satisfaction"])
        x_test = test.drop(columns=["satisfaction"])
        y_train = train["satisfaction"]
        y_test = test["satisfaction"]

        clf = MLPClassifier(hidden_layer_sizes=(25, 25, 24), activation='tanh', max_iter=1000,
                            tol=1e-5, solver='adam', learning_rate_init=0.001, verbose=False, random_state=42)
        clf.fit(x_train, y_train)
        y_test_assig = clf.predict(x_test)

        #Guardamos las métricas para este fold
        neuron_fold_accuracy.append(accuracy_score(y_test, y_test_assig))
        neuron_fold_f1.append(f1_score(y_test, y_test_assig, average='weighted'))
        neuron_fold_precision.append(precision_score(y_test, y_test_assig, average='weighted'))
        neuron_fold_recall.append(recall_score(y_test, y_test_assig, average='weighted'))
        neuron_fold_error.append(1 - accuracy_score(y_test, y_test_assig))
        neuron_train.append(clf.score(x_train, y_train))
        neuron_test.append(clf.score(x_test, y_test))

    #Guardamos las medias de cada métrica de la red neuronal
    neuron_accuracy = np.mean(neuron_fold_accuracy)
    neuron_f1 = np.mean(neuron_fold_f1)
    neuron_precision = np.mean(neuron_fold_precision)
    neuron_recall = np.mean(neuron_fold_recall)
    neuron_error = np.mean(neuron_fold_error)
    neuron_train = np.mean(neuron_train)
    neuron_test = np.mean(neuron_test)
    print("Neuronal Network accuracy: ", neuron_accuracy)
    print("Neuronal Network f1: ", neuron_f1)
    print("Neuronal Network precision: ", neuron_precision)
    print("Neuronal Network recall: ", neuron_recall)
    print("Neuronal Network error: ", neuron_error)
    print("Neuronal Network train: ", neuron_train)
    print("Neuronal Network test: ", neuron_test)

def generate_index_folds(num_rows:int, folds:int) -> np.ndarray:
    """Nos devuelve un array con los indices de los folds
    a los que pertenece cada fila"""
    tam_fold = num_rows // folds - 1 #Redondeo hacia abajo
    list_fold = []

    for i in range(folds):
        list_fold.extend([i] * tam_fold)

    list_fold.extend([folds - 1] * (num_rows - len(list_fold))) #-1 porque los indices empiezan en 0 

    return np.array(list_fold)

def create_folds(df:pd.DataFrame, folds:int) -> list:
    """Nos devolverá una lista de duplas donde cada dupla
    está formada por un dataframe de entrenamiento y otro de test"""
    df["fold"] = generate_index_folds(df.shape[0], folds)

    dupla_list = [(df[df["fold"] == i].drop(columns=["fold"]),
                   df[df["fold"] != i].drop(columns=["fold"]))
                   for i in range(folds)]

    return dupla_list

def kfold(df:pd.DataFrame, folds:int) -> list:
    """Está función realiza la validación con k-folds,
    dando una lista de tuplas con los dataframes de entrenamiento y test
    para cada fold"""
    df_rand = df.sample(frac=1).reset_index(drop=True)
    
    return create_folds(df_rand, folds)

"Data loading"
data = pd.read_csv('train_students.csv')

'''Creación de nuevas características'''
#1. Total Delay
data['Total Delay'] = data['Departure Delay in Minutes'] + data['Arrival Delay in Minutes']
data.drop(columns=['Departure Delay in Minutes', 'Arrival Delay in Minutes'], inplace=True)

#2. Age en ranges
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

'''Limpieza de Datos'''
imputer = KNNImputer(n_neighbors=5)
for col in data.columns:
    if data[col].isnull().sum() > 0:
        data[col] = imputer.fit_transform(data[[col]])

'''Transformación de datos'''
#Codificación variables categóricas
category_columns = [col for col in data.columns if data[col].dtype == 'object' or data[col].dtype == 'category']

for col in category_columns: #Ajustamos el LabelEncoder con todas las categorías posibles en los datos de entrenamiento
   le = LabelEncoder()
   data[col] = le.fit_transform(data[col])

#Normalización variables numéricas
numeric_columns = [col for col in data.columns if data[col].dtype != 'object' and col != 'satisfaction']

scaler = MinMaxScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

'''Separación en columnas características y etiquetas'''
X = data.drop(columns='satisfaction')
y = data['satisfaction']

"Selecting the best characteristics"
X_selected = select_kbest(X, y, 9)

#Concateno la columna de satisfacción, para poder entrenar el modelo
data = pd.concat([X, y], axis=1) #axis=1 para concatenar por columnas

"Making the neuronal network model"    
create_model(data, 5)