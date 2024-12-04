from sklearn.linear_model import LogisticRegression

ACCURACIES = {'Model': [], 'Accuracy': [], 'Train Score': [], 'Test Score': []}

def LR_model(df:pd.DataFrame, model:str) -> LogisticRegression:
    """Realiza el modelo de Regresión Logística y
    nos devuelve el modelo"""
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

    return LR

LR_stesc = LR_model(df_preprocessed_1, 'Standard Scaler')
LR_min = LR_model(df_preprocessed_2, 'Min-Max Scaler')
