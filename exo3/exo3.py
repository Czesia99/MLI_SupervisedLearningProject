import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Chargement des données
inputs = np.load('./inputs.npy')
labels = np.load('./labels.npy')

# Parse des données pour entraînements et tests
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

# Création et entraînement du modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# Prédictions avec les données de tests
predictions = model.predict(X_test)

# Evaluation du score du modèle
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: {:.2f}".format(accuracy))
