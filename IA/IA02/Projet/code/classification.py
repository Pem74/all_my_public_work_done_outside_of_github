import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import joblib

files_directory = './'
df = pd.read_csv("peptides_clustered2.csv")

X = df.drop(columns=["Actif", "Peptide"])
y = df["Actif"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

pipelines = {
    'knn': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]),
    'logreg': Pipeline([('scaler', StandardScaler()), ('logreg', LogisticRegression(max_iter=2000))]),
    'rf': Pipeline([('rf', RandomForestClassifier())]),
    'svm': Pipeline([('scaler', StandardScaler()), ('svm', SVC())]),
}

param_grids = {
    'knn': {
        'knn__n_neighbors': [1, 2, 3, 4],
        'knn__weights': ['uniform', 'distance']
    },
    'logreg': {
        'logreg__C': np.logspace(-3, 3, 7),
        'logreg__solver': ['liblinear', 'lbfgs']
    },
    'rf': {
        'rf__n_estimators': [10, 50, 100, 200, 300],
        'rf__max_depth': [None, 10, 20, 30]
    },
    'svm': {'svm__C': np.logspace(-3, 3, 7),
        'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
}

# Trouver les meilleurs hyperparamètres pour chaque modèle
best_models = {}
for model_name in pipelines:
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(pipelines[model_name], param_grids[model_name], cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")

# Évaluer les modèles sur l'ensemble de test
best_model_name = None
best_model = None
best_accuracy = 0
accuracies = {}

for model_name, model in best_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies[model_name] = accuracy
    print(f"Accuracy for {model_name}: {accuracy}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = model_name
        best_model = model

print(f"Best model is {best_model_name} with accuracy {best_accuracy}")

joblib.dump(best_model, "best_model_classification2.joblib")
print("Best model saved.")

plt.figure()
plt.bar(accuracies.keys(), accuracies.values(), color=['blue', 'green', 'red'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison')
plt.ylim(0, 1)
for i, v in enumerate(accuracies.values()):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
plt.savefig("classification_accuracy2.png")

# Charger le meilleur modèle pour une utilisation ultérieure
#best_model = joblib.load("best_model_classification.joblib")