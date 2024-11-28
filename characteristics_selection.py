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
 
#Selector pertenece a {'KBest', 'RFE', 'PCA'} y Normalization a {'Minmax', 'Standard'}
ACCURACIES = {'Selector':[], 'Accuracy':[], 'Normalization':[], 'Train accuracy':[], 'Test accuracy':[]}

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
    print(f"Accuracy of the model with {norm_type} and {selector} selector: {accuracy}")
    print(f"Test accuracy: {test_accuracy}")
    print(f"Train accuracy: {train_accuracy}")
    ACCURACIES['Accuracy'] = accuracy
    ACCURACIES['Train accuracy'] = train_accuracy
    ACCURACIES['Test accuracy'] = test_accuracy
    ACCURACIES['Normalization'] = norm_type
    ACCURACIES['Selector'] = selector


#Probamos los selectores de características con los conjuntos de datos de minmax y standard
data_original = pd.read_csv('train_students.csv')#Leemos el original para las etiquetas
y = data_original.iloc[:,-1].values

#Minmax
df_minmax = pd.read_csv('train_students_preprocessed_minmax.csv')
#Eliminamos la columna de etiquetas
df_minmax = df_minmax.drop(columns='satisfaction')

print('MinMax')
model_cs(select_kbest(df_minmax, y, 20), y, "MinMaxScaler", "SelectKBest(20)")
model_cs(select_kbest(df_minmax, y, 18), y, "MinMaxScaler", "SelectKBest(18)")
model_cs(select_rfe(df_minmax, y, 20), y, "MinMaxScaler", "RFE(20)")
model_cs(select_rfe(df_minmax, y, 18), y, "MinMaxScaler", "RFE(18)")
model_cs(select_pca(df_minmax, 20), y, "MinMaxScaler", "PCA(20)")
model_cs(select_pca(df_minmax, 16), y, "MinMaxScaler", "PCA(18)")

#Standard
df_standard = pd.read_csv('train_students_preprocessed_standard.csv')
#Eliminamos la columna de etiquetas
df_standard = df_standard.drop(columns='satisfaction')

print('Standard')
model_cs(select_kbest(df_standard, y, 20, f_classif), y, "StandardScaler", "SelectKBest(20)")
model_cs(select_kbest(df_standard, y, 18, f_classif), y, "StandardScaler", "SelectKBest(18)")
model_cs(select_rfe(df_standard, y, 20), y, "StandardScaler", "RFE(20)")
model_cs(select_rfe(df_standard, y, 18), y, "StandardScaler", "RFE(18)")
model_cs(select_pca(df_standard, 20), y, "StandardScaler", "PCA(20)")
model_cs(select_pca(df_standard, 16), y, "StandardScaler", "PCA(18)")

#Gráfica de SHAP
#Probamos los selectores de características con los conjuntos de datos de minmax y standard
df_min_max_best = select_kbest(df_minmax, y, 20)
df_standard_best = select_kbest(df_standard, y, 20, f_classif)

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
plt.suptitle("SHAP values for the best features selected by KBEST", fontsize=20)

plt.subplot(2, 1, 1)
#shap.plots.beeswarm(shap_values_minmax, max_display=16, show=False)
shap.summary_plot(shap_values_minmax, df_min_max_best, plot_type="bar", color="red", show=False)
plt.title("MinMaxScaler - Best 16 Features")

plt.subplot(2, 1, 2)
#shap.plots.beeswarm(shap_values_standard, max_display=16, show=False)
shap.summary_plot(shap_values_standard, df_standard_best, plot_type="bar", color="blue", show=False)
plt.title("StandardScaler - Best 16 Features")

plt.show()