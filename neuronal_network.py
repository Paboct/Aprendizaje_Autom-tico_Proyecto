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

'Dataset Loading'
data_original = pd.read_csv('train_students.csv') #para obtener la satisfacción
df_min_max = pd.read_csv('train_students_preprocessed_standard.csv') #Para obtener las x
df_min_max = df_min_max.drop(columns=['satisfaction'])

df_standard = pd.read_csv('train_students_preprocessed_minmax.csv') #Para obtener las x
df_standard = df_standard.drop(columns=['satisfaction'])

print("Minmax dataframe information")
print(df_min_max.info())

print("Standard dataframe information")
print(df_standard.info())

X = df_min_max.values[:, :-1]
y = data_original['satisfaction']

#Split train-test data
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
print('Precision: ', precision_score(y_test, y_test_assig_3, pos_label='neutral or dissatisfied'))
print('=======================')
print('Train Accuracy 2:',clf2.score(X_train, y_train))
print('Test Accuracy 2:',clf2.score(X_test, y_test))
print('Recall 2:', recall_score(y_test, y_test_assig_2, average='weighted'))
print('F1-score 2:', f1_score(y_test, y_test_assig_2, average='weighted'))
print('Error: ', 1 - accuracy_score(y_test, y_test_assig_2))
print('Precision 2: ', precision_score(y_test, y_test_assig_2, pos_label='neutral or dissatisfied'))
print('=======================')
#El pos_label es para indicar que clase se considera positiva, en este caso neutral