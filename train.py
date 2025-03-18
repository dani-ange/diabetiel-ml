import joblib
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("diabetes.csv")
X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save model
joblib.dump(model, 'diabetes_model.joblib')

# Save accuracy
with open("metrics.json", "w") as f:
    json.dump({"accuracy": accuracy * 100}, f)

print(f"Model Accuracy: {accuracy * 100}%")
