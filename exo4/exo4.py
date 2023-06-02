import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Chargement des données
inputs = np.load('./inputs.npy')
labels = np.load('./labels.npy')

# Parse données pour entraînements et test
X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.2, random_state=42)

# Création et entraînement du modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
predictions = model.predict(X_test)

# Evaluation du coefficient de détermination R2 du modèle
r2 = r2_score(y_test, predictions)
print("Coefficient de détermination R2 : {:.2f}".format(r2))