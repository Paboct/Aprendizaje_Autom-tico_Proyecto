import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt
 
def select_kbest(X:pd.DataFrame, y:np.array, ks:int, sc_func=chi2) -> pd.DataFrame:
    """Realiza la selección de las k mejores características, para un 
    dataframe de caracterísitcas y un array de etiqutas."""
    selector = SelectKBest(score_func=sc_func, k=ks)
    x_best = selector.fit_transform(X, y)
    best_columns = X.columns[selector.get_support()]

    return pd.DataFrame(x_best, columns=best_columns)

def select_rfe(X:pd.DataFrame, y:np.array, n_features:int) -> pd.DataFrame:
    """Realiza la selección de las n mejores características usando
    el Recursive Feature Elimination"""
    selector = RFE(estimator=LogisticRegression(), n_features_to_select=n_features)
    x_best = selector.fit_transform(X, y)
    best_columns = X.columns[selector.get_support()]

    return pd.DataFrame(x_best, columns=best_columns)

def select_pca(X:pd.DataFrame, n_comp:int) -> pd.DataFrame:
    """Realiza la selección de las n mejores características
     usando PCA"""
    selector = PCA(n_components=n_comp)
    x_best = selector.fit_transform(X)

    return pd.DataFrame(x_best, columns=["PCA{i}" for i in range(1, n_comp+1)])

def model_cs(X:pd.DataFrame, y:np.array, norm_type:str, selector:str='No') -> None:
    """Entrena un modelo de regresión logística y evalua su precisión"""
    LR = LogisticRegression(max_iter=100)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = LR.score(X_test, y_test)
    train_accuracy = LR.score(X_train, y_train)
    test_accuracy = LR.score(X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(f"Train accuracy: {train_accuracy}")
    print(f"Test accuracy: {test_accuracy}")
    print(f"Normalization: {norm_type}")
    print(f"Selector: {selector}")
    print()

#Probamos los selectores de características con los conjuntos de datos de minmax y standard
data_original = pd.read_csv('train_students.csv')#Leemos el original para las etiquetas
y = data_original.iloc[:,-1].values

#Minmax
df_minmax = pd.read_csv('train_students_preprocessed_minmax.csv')

#Eliminamos la columna de etiquetas
df_minmax = df_minmax.drop(columns='satisfaction')

print('MinMax')
model_cs(select_kbest(df_minmax, y, 10), y, "MinMaxScaler", "SelectKBest(10)")
model_cs(select_kbest(df_minmax, y, 9), y, "MinMaxScaler", "SelectKBest(9)")
model_cs(select_rfe(df_minmax, y, 10), y, "MinMaxScaler", "RFE(10)")
model_cs(select_rfe(df_minmax, y, 9), y, "MinMaxScaler", "RFE(9)")
model_cs(select_pca(df_minmax, 10), y, "MinMaxScaler", "PCA(10)")

#Standard
df_standard = pd.read_csv('train_students_preprocessed_standard.csv')
#Eliminamos la columna de etiquetas
df_standard = df_standard.drop(columns='satisfaction')

print('Standard')
model_cs(select_kbest(df_standard, y, 10, f_classif), y, "StandardScaler", "SelectKBest(10)")
model_cs(select_kbest(df_standard, y, 9, f_classif), y, "StandardScaler", "SelectKBest(9)")
model_cs(select_rfe(df_standard, y, 10), y, "StandardScaler", "RFE(10)")
model_cs(select_rfe(df_standard, y, 9), y, "StandardScaler", "RFE(9)")
model_cs(select_pca(df_standard, 10), y, "StandardScaler", "PCA(10)")

#Gráfica de SHAP
#Probamos los selectores de características con los conjuntos de datos de minmax y standard
df_min_max_best = select_kbest(df_minmax, y, 10)
df_standard_best = select_kbest(df_standard, y, 10, f_classif)

#Hacemos el modelo
model_min_max = LogisticRegression().fit(df_min_max_best, y)
model_standard = LogisticRegression().fit(df_standard_best, y)

#Hacmos el explainer
explainer_minmax = shap.Explainer(model_min_max, df_min_max_best)
explainer_standard = shap.Explainer(model_standard, df_standard_best)

#Calculamos los valores SHAP
shap_values_minmax = explainer_minmax(df_min_max_best)
shap_values_standard = explainer_standard(df_standard_best)

#Graficamos
plt.figure(figsize=(14, 16))
plt.suptitle("SHAP values for the best features selected by KBEST", fontsize=10)

plt.subplot(2, 1, 1)
shap.summary_plot(shap_values_minmax, df_min_max_best, plot_type="bar", color="red", show=False)
plt.title("MinMaxScaler - Best 10 Features")

plt.subplot(2, 1, 2)
shap.summary_plot(shap_values_standard, df_standard_best, plot_type="bar", color="blue", show=False)
plt.title("StandardScaler - Best 10 Features")

plt.show()