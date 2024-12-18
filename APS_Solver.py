import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
import joblib
import pickle
class APS_Solver:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.numeric_columns = []

    def load_model(self, model_path):
        """Load a pre-trained model from a file."""
        self.model, self.label_encoders, self.scaler = joblib.load(model_path)

    def save_model(self, model_path):
        """Save the trained model to a file."""
        with open(model_path, 'wb') as file:
            pickle.dump((self.model, self.label_encoders, self.scaler), file)

    def preprocess_data(self, df, is_training=True):
        """Preprocess the data for training or testing."""
        # Create new features
        df['Age Groups'] = pd.cut(df['Age'], bins=[0, 18, 35, 60, 100],
                                  labels=['Joven', 'Adulto Joven', 'Adulto', 'Adulto Mayor'])
        df['Distance Range'] = pd.cut(df['Flight Distance'], bins=[0, 500, 1500, 3000, 5000, 10000],
                                      labels=['Short', 'Medium', 'Long', 'Very Long', 'Ultra Long'])
        
        df['Total Delay'] = df['Arrival Delay in Minutes'] + df['Departure Delay in Minutes']

        df.drop(columns=['id','Age', 'Flight Distance', 'Arrival Delay in Minutes',
                         'Departure Delay in Minutes'], inplace=True)

        # Handle missing values
        imputer = KNNImputer(n_neighbors=5)
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col] = imputer.fit_transform(df[[col]])

        # Encode categorical variables
        category_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in category_columns:
            if is_training:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                self.label_encoders[col] = le
            else:
                df[col] = self.label_encoders[col].transform(df[col])

        # Normalize numeric columns
        if is_training:
            self.numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
            self.scaler = MinMaxScaler()
            df[self.numeric_columns] = self.scaler.fit_transform(df[self.numeric_columns])
        else:
            df[self.numeric_columns] = self.scaler.transform(df[self.numeric_columns])

        return df

    def train_model(self, file_path):
        """Train the model using the data from the provided file path."""
        df = pd.read_csv(file_path)
        df = self.preprocess_data(df, is_training=True)

        X = df.drop(columns='satisfaction')
        y = df['satisfaction']


        self.model = MLPClassifier(hidden_layer_sizes=(25, 25, 24), activation='tanh', max_iter=1000,
                                   tol=1e-5, solver='adam', learning_rate_init=0.001, random_state=42)

        self.model.fit(X, y)

    def test_model(self, file_path):
        """Test the model using the data from the provided file path."""
        df = pd.read_csv(file_path)
        df = self.preprocess_data(df, is_training=False)

        X = df.drop(columns='satisfaction')
        y_true = df['satisfaction']

        y_pred = self.model.predict(X)
        error_rate = 1 - accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"Tasa de error: {error_rate}")
        print(f"Precisión: {precision}")
        print(f"Recall: {recall}")
        print(f"F1-score: {f1}")
