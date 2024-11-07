import pandas as pd
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Load the dataset
df = pd.read_csv("train_new.csv")

# Let's examine the dataset 
df.head()
df.info()

# Shape of the dataset
print("Dataset shape:", df.shape)

# Visualizing the count of edible and poisonous mushrooms
print ("Examples of each class:", df['Transported'].value_counts())

#elimina los valores nulos de la ultima columna
df.dropna(subset=['Transported'], inplace=True)

#Encode the dataset
labelencoder=LabelEncoder()
for column in df.columns:
    df[column] = labelencoder.fit_transform(df[column])

print("muestra df", df)

# Create the label vector (y) and descriptor matrix (X)
X = df.drop('Transported', axis=1) #axis=1 eliminas columnas, y elimina la columna de transported
y = df['Transported']

print(X)

#Split train-test data
# Use 30% of the data for test and make the split stratified according to the class
# ====================== YOUR CODE HERE ======================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=1)

# Normalize the data using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



"""
# Train a logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = logreg.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Print the results
print("Accuracy regresio logistica: ", accuracy)
print("\nClassification Report:\n", classification_report_result)
"""

"""
# Train a Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = nb_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_result = classification_report(y_test, y_pred)

# Print the results
print("Accuracy GaussianNB:", accuracy)
print("\nClassification Report:\n", classification_report_result)
"""


#Create the classification model: Perceptron
# Indicate one hidden layer with 3 nodes in the hidden layer and activation function 'relu'
clf = MLPClassifier(hidden_layer_sizes=(3), tol=1e-5, activation='relu', solver='adam', verbose=True, random_state=1)

# Train the MLP
clf.fit(X_train, y_train)

# Compute the outputs for the test set
y_test_assig= clf.predict(X_test)

# Evaluate the Classifier 
# Compute the confusion matrix
cm = confusion_matrix(y_test_assig, y_test)

#muestra la matriz de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('Confusion Matrix ', fontsize=14)
plt.show()

print('=======================')
print('Train Accuracy:', clf.score(X_train, y_train))
print('Test Accuracy:', clf.score(X_test, y_test))
print('=======================')




