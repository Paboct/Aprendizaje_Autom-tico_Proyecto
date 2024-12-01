import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
#Importar recall 
from sklearn.metrics import recall_score
# Importar el f1-score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def select_kbest(X:pd.DataFrame, y:np.array, ks:int, sc_func=chi2) -> pd.DataFrame:
    """Realiza la selección de las k mejores características, para un 
    dataframe de caracterísitcas y un array de etiqutas."""
    selector = SelectKBest(score_func=sc_func, k=ks)
    x_best = selector.fit_transform(X, y)
    best_columns = X.columns[selector.get_support()]

    return pd.DataFrame(x_best, columns=best_columns)

def transform_data_min_max(df: pd.DataFrame, labels_encoded: dict, scaler:MinMaxScaler) -> pd.DataFrame:
    """
    Transforma las columnas categóricas y numéricas de un DataFrame con un LabelEncoder y MinMaxScaler.
    """

    for col, le in labels_encoded.items():
        df[col] = le.transform(df[col]) if df[col].iloc[0] in le.classes_ else -1  # Asignar -1 si es desconocido

    numeric_columns = [col for col in df.columns if col in scaler.feature_names_in_]
    df[numeric_columns] = scaler.transform(df[numeric_columns])

    return df

'Dataset Loading'
data_original = pd.read_csv('train_students.csv')

"""Creando un nuevo dataset"""
#Age en rangos
ranges = [0, 18, 35, 60, 100]
labels = ['Joven', 'Adulto Joven', 'Adulto', 'Adulto Mayor']

data_original['Age'] = pd.cut(data_original['Age'], bins=ranges, labels=labels)

#Flight Distance en rangos
ranges = [0, 500, 1500, 3000, 5000, 10000]
labels = ['Short', 'Medium', 'Long', 'Very Long', 'Ultra Long']

data_original['Distance Range'] = pd.cut(data_original['Flight Distance'], bins=ranges, labels=labels)

#Si elimino el id?
#data_original.drop(columns='id', inplace=True)

#Comfort Total
data_original['Comfort Total'] = data_original['Inflight wifi service'] + data_original['Departure/Arrival time convenient'] + data_original['Ease of Online booking'] + data_original['Gate location'] + data_original['Food and drink'] + data_original['Online boarding'] + data_original['Seat comfort'] + data_original['Inflight entertainment'] + data_original['On-board service'] + data_original['Leg room service'] + data_original['Baggage handling'] + data_original['Checkin service'] + data_original['Inflight service'] + data_original['Cleanliness']
data_original.drop(columns=['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'], inplace=True)

"""Limpiando los datos"""
imputer = KNNImputer(n_neighbors=5)
data_original['Arrival Delay in Minutes'] = imputer.fit_transform(data_original['Arrival Delay in Minutes'].values.reshape(-1, 1))
data_original['Arrival Delay in Minutes'] = data_original['Arrival Delay in Minutes'].astype(int)

"""Transformando los datos"""
#Codificación de variables categóricas
category_columns = [col for col in data_original.columns if data_original[col].dtype == 'object' or data_original[col].dtype == 'category']

#Inicializamos un diccionario para guardar los LabelEncoders de cada columna
labels_encoded = {}

for col in category_columns: #Ajustamos el LabelEncoder con todas las categorías posibles en los datos de entrenamiento
   le = LabelEncoder()
   data_original[col] = le.fit_transform(data_original[col])
   labels_encoded[col] = le

#Eliminamos la columna satisfaction del diccionario, ya que las predicciones no se harán sobre esta columna
labels_encoded.pop('satisfaction', None)

#Normalización de datos numéricos
numeric_columns = [col for col in data_original.columns if data_original[col].dtype != 'object' and col != 'satisfaction']

#Aplicamos Min-Max Scaler
min_max_scaler = MinMaxScaler()
data_original[numeric_columns] = min_max_scaler.fit_transform(data_original[numeric_columns])

X = data_original.drop(columns='satisfaction')
y = data_original['satisfaction']

#Podemos seleccioonara las mejores características
#X = select_kbest(X, y, 8)

#Split train-test data_original
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

#MLP training
clf3 = MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation='tanh', max_iter=1000,
                     tol=1e-4, solver='adam', learning_rate_init=0.01, verbose=False, random_state=42)
clf2 = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, activation='tanh', 
                     tol=1e-4, solver='adam',learning_rate_init=0.01, verbose=False, random_state=42)

#MLP training
clf3.fit(X_train, y_train)
clf2.fit(X_train, y_train)

#Predictions
y_test_assig_3 = clf3.predict(X_test)
y_test_assig_2 = clf2.predict(X_test)

#Evluador
cm3 = confusion_matrix(y_test, y_test_assig_3)
cm2 = confusion_matrix(y_test, y_test_assig_2)

disp = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=clf3.classes_)
disp.plot()
plt.title("MLP with 3 hidden layers", fontsize=20)
plt.show()

disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=clf2.classes_)
disp2.plot()
plt.title("MLP with 2 hidden layers", fontsize=20)
plt.show()

print(data_original['satisfaction'].value_counts())

print('=======================')
print('Train Accuracy:',clf3.score(X_train, y_train))
print('Test Accuracy:',clf3.score(X_test, y_test))
print('Recall:', recall_score(y_test, y_test_assig_3, average='weighted'))
print('F1-score:', f1_score(y_test, y_test_assig_3, average='weighted'))
print('Error: ', 1 - accuracy_score(y_test, y_test_assig_3))
#print('Precision: ', precision_score(y_test, y_test_assig_3, pos_label='neutral or dissatisfied'))
print('Precision: ', precision_score(y_test, y_test_assig_3, pos_label=0))
print('=======================')
print('Train Accuracy 2:',clf2.score(X_train, y_train))
print('Test Accuracy 2:',clf2.score(X_test, y_test))
print('Recall 2:', recall_score(y_test, y_test_assig_2, average='weighted'))
print('F1-score 2:', f1_score(y_test, y_test_assig_2, average='weighted'))
print('Error: ', 1 - accuracy_score(y_test, y_test_assig_2))
#print('Precision 2: ', precision_score(y_test, y_test_assig_2, pos_label='neutral or dissatisfied'))
print('Precision 2: ', precision_score(y_test, y_test_assig_2, pos_label=0))
print('=======================')
#El pos_label es para indicar que clase se considera positiva, en este caso neutral

'''0 es neutral or dissatisfied y 1 es satisfied'''
#print("Clases por valoresEncoded\n", data_original['satisfaction'].value_counts())
#print("Clases por valoresOriginal\n", pd.read_csv('train_students.csv')['satisfaction'].value_counts())

#Hacemos predicciones
example1 = {
    "id": 117930,
    "Gender": "Female",
    "Customer Type": "Loyal Customer",
    "Age": 25,
    "Type of Travel": "Business travel",
    "Class": "Business",
    "Flight Distance": 1716,
    "Inflight wifi service": 1,
    "Departure/Arrival time convenient": 1,
    "Ease of Online booking": 1,
    "Gate location": 1,
    "Food and drink": 2,
    "Online boarding": 2,
    "Seat comfort": 2,
    "Inflight entertainment": 2,
    "On-board service": 3,
    "Leg room service": 5,
    "Baggage handling": 4,
    "Checkin service": 5,
    "Inflight service": 5,
    "Cleanliness": 2,
    "Departure Delay in Minutes": 0,
    "Arrival Delay in Minutes": 0.0,
}

#"satisfaction": "satisfied"
example1 = pd.DataFrame([example1])
#Eliminamos la columna id, ya que no se utilizará para la predicción
#example1.drop(columns='id', inplace=True)

#Agregamos las columnas que se crearon
example1['Distance Range'] = pd.cut(example1['Flight Distance'], bins=ranges, labels=labels)
example1['Comfort Total'] = example1['Inflight wifi service'] + example1['Departure/Arrival time convenient'] + example1['Ease of Online booking'] + example1['Gate location'] + example1['Food and drink'] + example1['Online boarding'] + example1['Seat comfort'] + example1['Inflight entertainment'] + example1['On-board service'] + example1['Leg room service'] + example1['Baggage handling'] + example1['Checkin service'] + example1['Inflight service'] + example1['Cleanliness']
example1.drop(columns=['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'], inplace=True)

example_1 = transform_data_min_max(example1, labels_encoded, min_max_scaler)
prediction_clf3 = clf3.predict(example_1)
prediction_clf2 = clf2.predict(example_1)
print('=======================')
print("Prediction by cl3:")
print("neutral or dissatisfied" if prediction_clf3[0] == 0 else "satisfied")
print("Prediction by cl2:")
print("neutral or dissatisfied" if prediction_clf2[0] == 0 else "satisfied")

example2 = {
    "id": 43510,
    "Gender": "Female",
    "Customer Type": "Loyal Customer",
    "Age": 43,
    "Type of Travel": "Personal Travel",
    "Class": "Eco",
    "Flight Distance": 752,
    "Inflight wifi service": 3,
    "Departure/Arrival time convenient": 5,
    "Ease of Online booking": 3,
    "Gate location": 3,
    "Food and drink": 5,
    "Online boarding": 4,
    "Seat comfort": 5,
    "Inflight entertainment": 3,
    "On-board service": 3,
    "Leg room service": 3,
    "Baggage handling": 5,
    "Checkin service": 3,
    "Inflight service": 3,
    "Cleanliness": 4,
    "Departure Delay in Minutes": 52,
    "Arrival Delay in Minutes": 29.0
}

example2 = pd.DataFrame([example2])
#neutral or dissatisfied

#example2.drop(columns='id', inplace=True)

#Agregamos las columnas que se crearon
example2['Distance Range'] = pd.cut(example2['Flight Distance'], bins=ranges, labels=labels)
example2['Comfort Total'] = example2['Inflight wifi service'] + example2['Departure/Arrival time convenient'] + example2['Ease of Online booking'] + example2['Gate location'] + example2['Food and drink'] + example2['Online boarding'] + example2['Seat comfort'] + example2['Inflight entertainment'] + example2['On-board service'] + example2['Leg room service'] + example2['Baggage handling'] + example2['Checkin service'] + example2['Inflight service'] + example2['Cleanliness']
example2.drop(columns=['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'], inplace=True)

example_2 = transform_data_min_max(example2, labels_encoded, min_max_scaler)
prediction_clf3 = clf3.predict(example_2)
prediction_clf2 = clf2.predict(example_2)
print('=======================')
print("Prediction by cl3:")
print("neutral or dissatisfied" if prediction_clf3[0] == 0 else "satisfied")
print("Prediction by cl2:")
print("neutral or dissatisfied" if prediction_clf2[0] == 0 else "satisfied")

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

#Agregamos las columnas que se crearon
#example_3.drop(columns='id', inplace=True)
example_3['Distance Range'] = pd.cut(example_3['Flight Distance'], bins=ranges, labels=labels)
example_3['Comfort Total'] = example_3['Inflight wifi service'] + example_3['Departure/Arrival time convenient'] + example_3['Ease of Online booking'] + example_3['Gate location'] + example_3['Food and drink'] + example_3['Online boarding'] + example_3['Seat comfort'] + example_3['Inflight entertainment'] + example_3['On-board service'] + example_3['Leg room service'] + example_3['Baggage handling'] + example_3['Checkin service'] + example_3['Inflight service'] + example_3['Cleanliness']
example_3.drop(columns=['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'], inplace=True)

example_3 = transform_data_min_max(example_3, labels_encoded, min_max_scaler)
prediction_clf3 = clf3.predict(example_3)
prediction_clf2 = clf2.predict(example_3)
print('=======================')
print("Prediction by cl3:")
print("neutral or dissatisfied" if prediction_clf3[0] == 0 else "satisfied")
print("Prediction by cl2:")
print("neutral or dissatisfied" if prediction_clf2[0] == 0 else "satisfied")

example4 = {
    'id': 999999,
    'Gender': 'Male',
    'Customer Type': 'Loyal Customer',
    'Age': 35,
    'Type of Travel': 'Business travel',
    'Class': 'Business',
    'Flight Distance': 2000,
    'Inflight wifi service': 4,
    'Departure/Arrival time convenient': 4,
    'Ease of Online booking': 4,
    'Gate location': 3,
    'Food and drink': 5,
    'Online boarding': 5,
    'Seat comfort': 4,
    'Inflight entertainment': 5,
    'On-board service': 5,
    'Leg room service': 4,
    'Baggage handling': 5,
    'Checkin service': 5,
    'Inflight service': 5,
    'Cleanliness': 4,
    'Departure Delay in Minutes': 0,
    'Arrival Delay in Minutes': 0,
}

#sadisfied
example4 = pd.DataFrame([example4])
#example4.drop(columns='id', inplace=True)

#Agregamos las columnas que se crearon
example4['Distance Range'] = pd.cut(example4['Flight Distance'], bins=ranges, labels=labels)
example4['Comfort Total'] = example4['Inflight wifi service'] + example4['Departure/Arrival time convenient'] + example4['Ease of Online booking'] + example4['Gate location'] + example4['Food and drink'] + example4['Online boarding'] + example4['Seat comfort'] + example4['Inflight entertainment'] + example4['On-board service'] + example4['Leg room service'] + example4['Baggage handling'] + example4['Checkin service'] + example4['Inflight service'] + example4['Cleanliness']
example4.drop(columns=['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'], inplace=True)

example_4 = transform_data_min_max(example4, labels_encoded, min_max_scaler)
prediction_clf3 = clf3.predict(example_4)
prediction_clf2 = clf2.predict(example_4)
print('=======================')
print("Prediction by cl3:")
print("neutral or dissatisfied" if prediction_clf3[0] == 0 else "satisfied")
print("Prediction by cl2:")
print("neutral or dissatisfied" if prediction_clf2[0] == 0 else "satisfied")

example5 = {
    'id': 888888,
    'Gender': 'Female',
    'Customer Type': 'disloyal Customer',
    'Age': 28,
    'Type of Travel': 'Personal Travel',
    'Class': 'Eco Plus',
    'Flight Distance': 500,
    'Inflight wifi service': 2,
    'Departure/Arrival time convenient': 2,
    'Ease of Online booking': 3,
    'Gate location': 4,
    'Food and drink': 3,
    'Online boarding': 2,
    'Seat comfort': 3,
    'Inflight entertainment': 2,
    'On-board service': 3,
    'Leg room service': 2,
    'Baggage handling': 3,
    'Checkin service': 4,
    'Inflight service': 3,
    'Cleanliness': 3,
    'Departure Delay in Minutes': 30,
    'Arrival Delay in Minutes': 20,
}

#neutral or dissatisfied
example5 = pd.DataFrame([example5])
#example5.drop(columns='id', inplace=True)
example5['Distance Range'] = pd.cut(example5['Flight Distance'], bins=ranges, labels=labels)
example5['Comfort Total'] = example5['Inflight wifi service'] + example5['Departure/Arrival time convenient'] + example5['Ease of Online booking'] + example5['Gate location'] + example5['Food and drink'] + example5['Online boarding'] + example5['Seat comfort'] + example5['Inflight entertainment'] + example5['On-board service'] + example5['Leg room service'] + example5['Baggage handling'] + example5['Checkin service'] + example5['Inflight service'] + example5['Cleanliness']
example5.drop(columns=['Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness'], inplace=True)

example_5 = transform_data_min_max(example5, labels_encoded, min_max_scaler)
prediction_clf3 = clf3.predict(example_5)
prediction_clf2 = clf2.predict(example_5)
print('=======================')
print("Prediction by cl3:")
print("neutral or dissatisfied" if prediction_clf3[0] == 0 else "satisfied")
print("Prediction by cl2:")
print("neutral or dissatisfied" if prediction_clf2[0] == 0 else "satisfied")
print('=======================')