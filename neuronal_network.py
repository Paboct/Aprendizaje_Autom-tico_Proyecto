import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

'Dataset Loading'
data_original = pd.read_csv('train_students.csv') #para obtener la satisfacción
df_min_max = pd.read_csv('train_students_preprocessed_standard.csv') #Para obtener las x
df_min_max = df_min_max.drop(columns=['satisfaction'])

df_standard = pd.read_csv('train_students_preprocessed_minmax.csv') #Para obtener las x
df_standard = df_standard.drop(columns=['satisfaction'])

X_min_max = df_min_max.values[:, :-1]
X_standard = df_min_max.values[:, :-1]

y = data_original['satisfaction']