from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score, precision_score ,recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from characteristics_selection import select_kbest, select_rfe, select_pca
from sklearn.datasets import make_moons
import pandas as pd
import numpy as np

def create_models_df(k_list:list, data:pd.DataFrame, n_folds:int=5) -> list:
    """Devolverá una lista de los dataframe que contine la accurcy, f1-score, precision y recall de
    los distintos modelos a probar."""
    #Dataframes de los modelos
    df_knn = {"K_neighbours":k_list}
    df_neuron = {}
    df_dt = {}

    #Listas de los accuracies de cada modelo
    knn_accuracy, neuron_accuracy, dt_accuracy = [], [], []
    knn_f1, neuron_f1, dt_f1 = [], [], []
    knn_precision, neuron_precision, dt_precision = [],  [], []
    knn_recall, neuron_recall, dt_recall = [], [], []
    knn_error, neuron_error, dt_error = [], [], []
    neuron_train, neuron_test = [], []

    #Creamos los folds para realizar los entrenamientos
    k_fold_list = kfold(data, n_folds)

    """KNN"""
    for k in k_list:
        #Listas de las métricas de cada fold
        knn_fold_accuracy = []
        knn_fold_f1 = []
        knn_fold_precision = []
        knn_fold_recall = []
        knn_fold_error = []

        for train, test in k_fold_list:
            # Hacemos la división de los datos
            x_train = train.drop(columns=["satisfaction"])
            y_train = train["satisfaction"]
            x_test = test.drop(columns=["satisfaction"])
            y_test = test["satisfaction"]

            """KNN"""
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            y_pred = knn.predict(x_test)
            
            #Guardamos las métricas para esté fold
            knn_fold_accuracy.append(accuracy_score(y_test, y_pred))
            knn_fold_f1.append(f1_score(y_test, y_pred, average='weighted'))
            knn_fold_precision.append(precision_score(y_test, y_pred, average='weighted'))
            knn_fold_recall.append(recall_score(y_test, y_pred, average='weighted'))
            knn_fold_error.append(1 - accuracy_score(y_test, y_pred))

        #Guardamos las medias de cada métrica de KNN para cada k
        knn_accuracy.append(np.mean(knn_fold_accuracy))
        knn_f1.append(np.mean(knn_fold_f1))
        knn_precision.append(np.mean(knn_fold_precision))
        knn_recall.append(np.mean(knn_fold_recall))
        knn_error.append(np.mean(knn_fold_error))



    """Neuronal Network"""
    #Creamos las listas de las métricas de cada fold
    neuron_fold_accuracy = []
    neuron_fold_f1 = []
    neuron_fold_precision = []
    neuron_fold_recall = []
    neuron_fold_error = []
    train_fold = []
    test_fold = []

    #Si queremos que la red neuronal use make_moons
    #data = make_moons(n_samples=data.shape[0], noise=0.3, random_state=42)
    #data = pd.DataFrame(data, columns=['x', 'y'])
    #k_fold_list = kfold(data, n_folds)

    for train, test in k_fold_list:

        clf = MLPClassifier(hidden_layer_sizes=(9, 9, 9), activation='tanh', max_iter=1000,
                            tol=1e-5, solver='adam', learning_rate_init=0.001, verbose=True, random_state=42)
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
    neuron_accuracy.append(np.mean(neuron_fold_accuracy) * 100)
    neuron_f1.append(np.mean(neuron_fold_f1) * 100)
    neuron_precision.append(np.mean(neuron_fold_precision) * 100)
    neuron_recall.append(np.mean(neuron_fold_recall) * 100)
    neuron_error.append(np.mean(neuron_fold_error) * 100)
    neuron_train = np.mean(neuron_train)
    neuron_test = np.mean(neuron_test)
    print("Neuronal Network accuracy: ", neuron_accuracy)
    print("Neuronal Network f1: ", neuron_f1)
    print("Neuronal Network precision: ", neuron_precision)
    print("Neuronal Network recall: ", neuron_recall)
    print("Neuronal Network error: ", neuron_error)
    print("Neuronal Network train: ", neuron_train)
    print("Neuronal Network test: ", neuron_test)


    """Decision Tree"""
    #Creamos las listas de las métricas de cada fold
    dt_fold_accuracy = []
    dt_fold_f1 = []
    dt_fold_precision = []
    dt_fold_recall = []
    dt_fold_error = []

    for train, test in k_fold_list:
        dt = DecisionTreeClassifier()
        dt.fit(x_train, y_train)
        y_pred = dt.predict(x_test)

        #Guardamos las métricas para este fold
        dt_fold_accuracy.append(accuracy_score(y_test, y_pred))
        dt_fold_f1.append(f1_score(y_test, y_pred, average='weighted'))
        dt_fold_precision.append(precision_score(y_test, y_pred, average='weighted'))
        dt_fold_recall.append(recall_score(y_test, y_pred, average='weighted'))
        dt_fold_error.append(1 - accuracy_score(y_test, y_pred))

    #Guardamos las medias de cada métrica del decision tree
    dt_accuracy.append(np.mean(dt_fold_accuracy) * 100)
    dt_f1.append(np.mean(dt_fold_f1) * 100)
    dt_precision.append(np.mean(dt_fold_precision) * 100)
    dt_recall.append(np.mean(dt_fold_recall) * 100)
    dt_error.append(np.mean(dt_fold_error) * 100)

    #Actualizamos los diccionarios
    """KNN"""
    df_knn["Accuracy"] = knn_accuracy
    df_knn["F1-score"] = knn_f1
    df_knn["Precision"] = knn_precision
    df_knn["Recall"] = knn_recall
    df_knn["Error"] = knn_error

    """Neuronal Network"""
    df_neuron["Accuracy"] = neuron_accuracy
    df_neuron["F1-score"] = neuron_f1
    df_neuron["Precision"] = neuron_precision
    df_neuron["Recall"] = neuron_recall
    df_neuron["Error"] = neuron_error


    """Decision Tree"""
    df_dt["Accuracy"] = dt_accuracy
    df_dt["F1-score"] = dt_f1
    df_dt["Precision"] = dt_precision
    df_dt["Recall"] = dt_recall
    df_dt["Error"] = dt_error


    #Creamos los dataframes
    df_knn = pd.DataFrame(df_knn, columns=[col for col in df_knn.keys()])
    df_neuron = pd.DataFrame(df_neuron, columns=[col for col in df_neuron.keys()])
    df_dt = pd.DataFrame(df_dt, columns=[col for col in df_dt.keys()])

    return (df_knn, df_neuron, df_dt)


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
#Cargamos la columna de satisfacción
original_dataset = pd.read_csv('train_students.csv')
y = original_dataset.values[:,-1]
print(y)
#Cargamos los dataframes preprocesados
df_min_max = pd.read_csv('train_students_preprocessed_minmax.csv')
df_min_max = df_min_max.drop(columns='satisfaction')

#Cargamos los dataframes preprocesados
#df_standard = pd.read_csv('train_students_preprocessed_standard.csv')
#df_standard.drop(columns=['satisfaction'])

"Selecting the best characteristics"
#kbest
print(df_min_max.columns)
data = select_kbest(df_min_max, y, 10)
print(data.columns)
#Concateno la columna de satisfacción, para poder entrenar el modelo
data = pd.concat([data, original_dataset['satisfaction']], axis=1) #axis=1 para concatenar por columnas

#RFE
#data = select_rfe(df_min_max, y, 12)
#data = pd.concat([data, original_dataset['satisfaction']], axis=1) #axis=1 para concatenar por columnas

#PCA
#data = select_pca(df_min_max, 12)
#data = pd.concat([data, original_dataset['satisfaction']], axis=1) #axis=1 para concatenar por columnas

"Creating the models dataframe"
k_list = [1, 3, 5, 7, 9, 11]
knn, neuronal_network, dt = create_models_df(k_list, data, 5)

#Guardamos el dataframe en un archivo csv
knn.to_csv('knn_metrics.csv', index=False)
neuronal_network.to_csv('neuronal_network_metrics.csv', index=False)
dt.to_csv('dt_metrics.csv', index=False)