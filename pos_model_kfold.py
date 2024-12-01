from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.impute import KNNImputer
import numpy as np
import pandas as pd

# Funciones auxiliares para K-Fold
def generate_index_folds(num_rows: int, folds: int) -> np.ndarray:
    """Genera un array con los índices de los folds."""
    tam_fold = num_rows // folds - 1  # Redondeo hacia abajo
    list_fold = []
    for i in range(folds):
        list_fold.extend([i] * tam_fold)
    list_fold.extend([folds - 1] * (num_rows - len(list_fold)))  # Completa con el último fold
    return np.array(list_fold)

def create_folds(df: pd.DataFrame, folds: int) -> list:
    """Genera una lista de pares de DataFrames (entrenamiento, test) por cada fold."""
    df["fold"] = generate_index_folds(df.shape[0], folds)
    dupla_list = [(df[df["fold"] != i].drop(columns=["fold"]),
                   df[df["fold"] == i].drop(columns=["fold"]))
                  for i in range(folds)]
    return dupla_list

def kfold(df: pd.DataFrame, folds: int) -> list:
    """Realiza validación cruzada con k-folds."""
    df_rand = df.sample(frac=1).reset_index(drop=True)  # Aleatoriza los datos
    return create_folds(df_rand, folds)

# Carga de datos
data_original = pd.read_csv('train_students.csv')

# Preprocesamiento
# Agrupación de edades
ranges = [0, 18, 35, 60, 100]
labels = ['Joven', 'Adulto Joven', 'Adulto', 'Adulto Mayor']
data_original['Age'] = pd.cut(data_original['Age'], bins=ranges, labels=labels)

# Agrupación de distancias
ranges_distance = [0, 500, 1500, 3000, 5000, 10000]
labels_distance = ['Short', 'Medium', 'Long', 'Very Long', 'Ultra Long']
data_original['Distance Range'] = pd.cut(data_original['Flight Distance'], bins=ranges_distance, labels=labels_distance)

# Crear columna "Comfort Total"
data_original['Comfort Total'] = (
    data_original['Inflight wifi service'] +
    data_original['Departure/Arrival time convenient'] +
    data_original['Ease of Online booking'] +
    data_original['Gate location'] +
    data_original['Food and drink'] +
    data_original['Online boarding'] +
    data_original['Seat comfort'] +
    data_original['Inflight entertainment'] +
    data_original['On-board service'] +
    data_original['Leg room service'] +
    data_original['Baggage handling'] +
    data_original['Checkin service'] +
    data_original['Inflight service'] +
    data_original['Cleanliness']
)
data_original.drop(columns=['Inflight wifi service', 'Departure/Arrival time convenient',
                            'Ease of Online booking', 'Gate location', 'Food and drink',
                            'Online boarding', 'Seat comfort', 'Inflight entertainment',
                            'On-board service', 'Leg room service', 'Baggage handling',
                            'Checkin service', 'Inflight service', 'Cleanliness'], inplace=True)

# Imputar valores faltantes
imputer = KNNImputer(n_neighbors=5)
data_original['Arrival Delay in Minutes'] = imputer.fit_transform(
    data_original['Arrival Delay in Minutes'].values.reshape(-1, 1))
data_original['Arrival Delay in Minutes'] = data_original['Arrival Delay in Minutes'].astype(int)

# Codificación de variables categóricas
category_columns = [col for col in data_original.columns if data_original[col].dtype == 'object' or data_original[col].dtype.name == 'category']
labels_encoded = {}
for col in category_columns:
    le = LabelEncoder()
    data_original[col] = le.fit_transform(data_original[col])
    labels_encoded[col] = le

# Normalización global de datos numéricos
numeric_columns = [col for col in data_original.columns if data_original[col].dtype != 'object' and col != 'satisfaction']
min_max_scaler = MinMaxScaler()
data_original[numeric_columns] = min_max_scaler.fit_transform(data_original[numeric_columns])

# Dividir X e y
X = data_original.drop(columns='satisfaction')
y = data_original['satisfaction']
data = pd.concat([X, y], axis=1)

# K-fold y entrenamiento
neuron_fold_accuracy = []
neuron_fold_f1 = []
neuron_fold_precision = []
neuron_fold_recall = []
neuron_fold_error = []

for train, test in kfold(data, 5):
    # Entrenamiento del modelo
    clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), activation='tanh', max_iter=1000,
                        tol=1e-4, solver='adam', learning_rate_init=0.001, verbose=True, random_state=42)
    clf.fit(train.drop(columns='satisfaction'), train['satisfaction'])
    
    # Predicciones
    y_test_assig = clf.predict(test.drop(columns='satisfaction'))
    
    # Cálculo de métricas
    neuron_fold_accuracy.append(accuracy_score(test['satisfaction'], y_test_assig))
    neuron_fold_f1.append(f1_score(test['satisfaction'], y_test_assig, average='weighted'))
    neuron_fold_precision.append(precision_score(test['satisfaction'], y_test_assig, average='weighted'))
    neuron_fold_recall.append(recall_score(test['satisfaction'], y_test_assig, average='weighted'))
    neuron_fold_error.append(1 - accuracy_score(test['satisfaction'], y_test_assig))

# Resultados finales
neuron_accuracy = np.mean(neuron_fold_accuracy) * 100
neuron_f1 = np.mean(neuron_fold_f1) * 100
neuron_precision = np.mean(neuron_fold_precision) * 100
neuron_recall = np.mean(neuron_fold_recall) * 100
neuron_error = np.mean(neuron_fold_error) * 100

print(f"Neuronal Network accuracy: {neuron_accuracy:.2f}%")
print(f"Neuronal Network f1: {neuron_f1:.2f}%")
print(f"Neuronal Network precision: {neuron_precision:.2f}%")
print(f"Neuronal Network recall: {neuron_recall:.2f}%")
print(f"Neuronal Network error: {neuron_error:.2f}%")

# Funciones para predicciones
def transform_example(example: dict, labels_encoded: dict, min_max_scaler, numeric_columns: list, ranges_age, labels_age, ranges_distance, labels_distance) -> pd.DataFrame:
    example_df = pd.DataFrame([example])
    
    # Agrupación de edades y distancias
    example_df['Age'] = pd.cut(example_df['Age'], bins=ranges_age, labels=labels_age)
    example_df['Distance Range'] = pd.cut(example_df['Flight Distance'], bins=ranges_distance, labels=labels_distance)
    
    # Calcular "Comfort Total"
    comfort_columns = [
        'Inflight wifi service', 'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment',
        'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service',
        'Inflight service', 'Cleanliness'
    ]
    example_df['Comfort Total'] = example_df[comfort_columns].sum(axis=1)
    example_df.drop(columns=comfort_columns, inplace=True)
    
    # Codificar columnas categóricas
    category_columns = [col for col in example_df.columns if example_df[col].dtype == 'object' or example_df[col].dtype.name == 'category']
    for col in category_columns:
        if col in labels_encoded:
            example_df[col] = example_df[col].map(
                lambda x: labels_encoded[col].classes_.tolist().index(x) if x in labels_encoded[col].classes_ else -1
            )
            if (example_df[col] == -1).any():
                raise ValueError(f"Nueva categoría encontrada en '{col}': {example_df[col][example_df[col] == -1]}")
    
    # Escalar columnas numéricas
    numeric_columns_example = [col for col in numeric_columns if col in example_df.columns]
    example_df[numeric_columns_example] = min_max_scaler.transform(example_df[numeric_columns_example])
    
    return example_df

def make_prediction(model, example: dict):
    try:
        example_df = transform_example(
            example, labels_encoded, min_max_scaler, numeric_columns, 
            ranges, labels, ranges_distance, labels_distance
        )
        prediction = model.predict(example_df)
        return "Satisfied" if prediction[0] == 1 else "Neutral or Dissatisfied"
    except ValueError as e:
        print(f"Error en la transformación del ejemplo: {e}")
        return "Error al predecir"

# Ejemplos para predicción
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

# Predicciones para los ejemplos
example1_prediction = make_prediction(clf, example1)
example2_prediction = make_prediction(clf, example2)
example4_prediction = make_prediction(clf, example4)

print(f"Predicción para el ejemplo 1: {example1_prediction}")
print(f"Predicción para el ejemplo 2: {example2_prediction}")
print(f"Predicción para el ejemplo 4: {example4_prediction}")
