import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

file = 'excel_limpio.csv' #ruta donde tengas descargado el dataset
data = pd.read_csv(file)
num_rows=data.shape[0]
cols=data.shape[1]
folds=5

def generate_index_folds(num_rows,folds):
    tam_fold = num_rows // folds
    lista_fold = []

    for i in range(folds):
        for j in range(tam_fold):
            lista_fold.append(i)

    lista_fold.extend([folds-1]*(num_rows - len(lista_fold)))

    return np.array(lista_fold)
folds = 5
data_rand = data.sample(frac=1).reset_index(drop=True)
print(generate_index_folds(data_rand.shape[0], folds))


def create_folds(data_rand, folds):
    dupla_list = [(data_rand[data_rand['fold'] != i], data_rand[data_rand['fold'] == i]) for i in range(folds)]
    return dupla_list

def kfold(data, folds):
    data_rand = data.sample(frac=0.5).reset_index(drop=True) #truco para desordenar los datps
    data_rand["fold"] = generate_index_folds(data_rand.shape[0], folds)
    dupla_list = create_folds(data_rand, folds)
    return dupla_list
'''
def train_knn_with_kfold(data, folds):
    accuracies = []

    for train, test in kfold(data, folds):
        model = KNeighborsClassifier(n_neighbors=3)  # Puedes ajustar el número de vecinos según sea necesario
        y_train = train['Outcome']
        x_train = train.drop('Outcome', axis=1)
        y_test = test['Outcome']
        x_test = test.drop('Outcome', axis=1)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

    acc_df = pd.DataFrame(accuracies, columns=["KNN"]).T
    acc_df["mean"] = acc_df.mean(axis=1)
    acc_df["std"] = acc_df.std(axis=1)
    print(acc_df)
'''


def train_decision_tree_with_kfold(data, folds):
    accuracies_knn = []
    accuracies_dt = []
    accuracies = []
    
    for train, test in kfold(data, folds):
        # KNN Model
        knn_model = KNeighborsClassifier(n_neighbors=3)
        y_train_knn = train['Transported']
        x_train_knn = train.drop('Transported', axis=1)
        y_test_knn = test['Transported']
        x_test_knn = test.drop('Transported', axis=1)

        scaler = StandardScaler()
        x_train_scaled_knn = scaler.fit_transform(x_train_knn)
        x_test_scaled_knn = scaler.transform(x_test_knn)#NO SE HACE FIT PORQUE SE PERDERIA EL ENTRENAMIENTO DEL MODELO
        
        knn_model.fit(x_train_scaled_knn, y_train_knn)
        #APARTADO ARBOL
        y_pred_knn = knn_model.predict(x_test_scaled_knn)
        accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
        accuracies_knn.append(accuracy_knn)

        # Decision Tree Model
        dt_model = DecisionTreeClassifier(random_state=42)

        dt_model.fit(x_train_scaled_knn, y_train_knn)
        y_pred_dt = dt_model.predict(x_test_scaled_knn)
        accuracy_dt = accuracy_score(y_test_knn, y_pred_knn)
        accuracies_dt.append(accuracy_dt)
        
        #APARTADO KNN WITH KFOLD
        #knn_model.fit(x_train_scaled_knn, y_train_knn)
        y_pred = knn_model.predict(x_test_scaled_knn)
        
        accuracy = accuracy_score(y_test_knn, y_pred)
        accuracies.append(accuracy)
    #APARTADO DATFRAME ARBOL    
    acc_df = pd.DataFrame({'KNN': accuracies_knn, 'DecisionTree': accuracies_dt})
    acc_df['Mean_KNN'] = acc_df['KNN'].mean()
    acc_df['Mean_DT'] = acc_df['DecisionTree'].mean()

    plt.bar(['KNN', 'Decision Tree'], [acc_df['Mean_KNN'].values[0], acc_df['Mean_DT'].values[0]])
    plt.title('Mean Accuracy Comparison between KNN and Decision Tree')
    plt.xlabel('Models')
    plt.ylabel('Mean Accuracy')
    plt.show()

    print(acc_df)
    #APARTADO KNN WITH KFOLD DATAFRAME
    acc_df = pd.DataFrame(accuracies, columns=["KNN"]).T
    acc_df["mean"] = acc_df.mean(axis=1)
    acc_df["std"] = acc_df.std(axis=1)
    print(acc_df)
# Ejemplo de uso
train_decision_tree_with_kfold(data, folds=5)
#train_knn_with_kfold(data, folds=5)

'''
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Cargar el conjunto de datos en un array
array = data.values

# Dividir el array en características (X) y variable objetivo (Y)
X = array[:, 0:8]
Y = array[:, 8]

# Especificar el número de folds y la semilla para la reproducibilidad
num_folds = 10
seed = 7

# Crear un objeto KFold para realizar la validación cruzada (KFold)
kf = model_selection.KFold(num_folds, random_state=seed, shuffle=True)

# Crear un modelo de Support Vector Classifier (SVC)
model = SVC()

# Evaluar el modelo utilizando la validación cruzada y obtener los resultados
result = cross_val_score(model, X, Y, cv=kf)

# Imprimir la precisión media del clasificador SVM
print("SVM Classifier Accuracy: %.3f" % (result.mean() * 100))


En resumen:

Cargar Datos: El conjunto de datos se carga en un array de NumPy.
Separar Características y Variable Objetivo: Se separan las características (X) de la variable objetivo (Y).
Configurar Parámetros de Validación Cruzada: Se especifica el número de folds y la semilla para garantizar la reproducibilidad.
Crear Objeto KFold: Se crea un objeto KFold que se utilizará para realizar la validación cruzada. Los datos se mezclan (shuffle=True) antes de dividirlos en folds.
Crear Modelo: Se crea un modelo de Support Vector Classifier (SVC).
Validación Cruzada y Evaluación del Modelo: Se utiliza la función cross_val_score para realizar la validación cruzada y obtener la precisión para cada fold.
Imprimir Resultados: Se imprime la precisión media del clasificador SVM en porcentaje.
'''
