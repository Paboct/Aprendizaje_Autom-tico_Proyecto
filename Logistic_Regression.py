from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt


def LR_model(df:pd.DataFrame, model:str) -> pd.DataFrame:
    """Realiza el modelo de Regresión Logística y
    nos devuelve el dataframe con las métricas"""

    #Diccionario con las métricas
    ACCURACIES = {'Model': [], 'Accuracy': [], 'Train Score': [], 'Test Score': []}
    
    X = df.drop(columns=['satisfaction'])
    y = df['satisfaction']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    LR = LogisticRegression()
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}\n")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LR.classes_)
    disp.plot()
    plt.title(f"Confusion Matrix {model}")
    
    train_score = LR.score(X_train, y_train)
    test_score = LR.score(X_test, y_test)

    print(f"Train Accuracy Score: {train_score:.4f}")
    print(f"Test Accuracy Score: {test_score:.4f}")

    ACCURACIES['Model'].append(model)
    ACCURACIES['Accuracy'].append(accuracy)
    ACCURACIES['Train Score'].append(train_score)
    ACCURACIES['Test Score'].append(test_score)

    return pd.DataFrame(ACCURACIES, columns=[i for i in ACCURACIES.keys()])

"Cargamos los datasets preprocesados"
df_standard = pd.read_csv('train_students_preprocessed_standard.csv')
df_minmax = pd.read_csv('train_students_preprocessed_minmax.csv')

# Eliminamos la columna de etiquetas en ambos datasets
X_min_max = df_minmax.drop(columns=['satisfaction'])
y_min_max = df_minmax['satisfaction']

#Realizamos el modelo de Regresión Logística
df_LR_stesc = LR_model(df_standard, 'Standard Scaler')
df_LR_minesc = LR_model(df_minmax, 'Min-Max Scaler')

#Creamos el csv con las métricas
df_LR_stesc.to_csv('LR_standard_scaler.csv', index=False)
df_LR_minesc.to_csv('LR_minmax_scaler.csv', index=False)