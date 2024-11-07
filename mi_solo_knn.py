import numpy as np
from collections import Counter
#librerias del paso 4 (scikit-learn)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
#PASO 1 CONSTRUCTOR
class KNN:
    def __init__(self,k,problem,metric):
        self.k=k #vecinos
        self.problem=problem #reg o clas (regresion o clasificacion)
        self.metric=metric #distancia euc o men
        #self.x_train = None
        #self.y_train = None
    #PASO 2 FIT (almacena los datos de entrenamiento para hacer luego la prediccion)
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
    #PASO 3 PREDECIR
    #DISTANCIAS
    def _euclidean_distance(self,x):
        distances = np.sqrt(np.sum((self.x_train - x) ** 2, axis=1))
        return distances
    def _manhattan_distance(self,x):
        distances = np.sum(np.abs(self.x_train - x), axis=1)
        return distances
    
    def _get_k_index(self,distances):#DEVUELVE LOS K INDICES CON LA DISTANCIA MAS PEQUEÑA
        return np.argsort(distances)[:self.k]    
    def _determine_class(self,indexes):#DADOS LOS INDICES (k_index) DEVELVE LA CLASE A LA QUE PERTENECE EL ELEMENTO (MAYORITARIA)
        classes = self.y_train[indexes]
        most_common = Counter(classes).most_common()
        return most_common[0][0]
    def _determine_value(self, indexes):#DEVUELVE EL VALOR MEDIO DE TODOS LOS VALORES DE LAS  ETIQUETAS VECINAS
        return np.mean(self.y_train[indexes])

    def predict(self,x_test): #devuelve la clase ala que pertence x_test
        if self.metric == "euc":
            distances = self._euclidean_distance(x_test)
        else:
            distances = self._manhattan_distance(x_test)

        indexes = self._get_k_index(distances)
        if self.problem == "reg":
            return self._determine_value(indexes)
        else:
            return self._determine_class(indexes)
#PASO 4 PROBAR LA LIBRERIA SCIKIT-LEARN

df = pd.read_csv('excel_limpio.csv')

# Supongamos que tienes una columna llamada 'target' que contiene las etiquetas
# Reemplaza 'target' con el nombre real de tu columna de etiquetas
X = df.drop('Transported', axis=1)  # Elimina la columna de etiquetas para obtener las características
y = df['Transported']  # La columna de etiquetas

# Dividir los datos en conjuntos de entrenamiento y prueba
# El parámetro test_size determina la proporción de datos que se asignarán al conjunto de prueba
# El parámetro random_state garantiza reproducibilidad
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#ESTE PASO HAY QUE HACER???????
# NORMALIZAMOS LOS DATOS
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)#NO SE HACE FIT PORQUE SE PERDERIA EL ENTRENAMIENTO DEL MODELO

#Cuando llames al método KNeighborsClassifier debes pasarle como número de vecinos 3, y métrica Manhattan
knn_sklearn = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn_sklearn.fit(x_train_scaled, y_train)
y_pred_sklearn = knn_sklearn.predict(x_test_scaled)

# CALCULAMOS ACURACY CON Scikit-Learn KNeighborsClassifier
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
#Para evaluar el resultado de tu clasificación puedes usar:
print("Scikit-Learn KNN Accuracy: ", 100 * accuracy_sklearn, "%")

# UTILIZAMOS LA CLASE HECHA
#Para calcular el accuracy puedes hacer uso de la función “accuracy_score(y_test,y_pred)”.

knn = KNN(3, problem="clas", metric="man")
knn.fit(x_train_scaled, y_train)
#DESPUES DE HACER EL FIT USAMOS EL PREDICT CON LOS DATOS DE TESTS CON LA SIGUINETE LINEA
#Después de hacer el fit, puedes usar el predict con los datos de test de la siguiente manera:
y_pred_custom = [knn.predict(x) for x in x_test_scaled]

# Calculate accuracy using the custom KNN class
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print("Custom KNN Accuracy: ", 100 * accuracy_custom, "%")
