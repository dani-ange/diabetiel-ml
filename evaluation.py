import joblib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset again (same as in train.py)
data = pd.read_csv("diabetes.csv")
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split data (same as in train.py)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
model = joblib.load('diabetes_model.joblib')

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Save accuracy to a JSON file
metrics = {"accuracy": accuracy * 100}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

# Print accuracy (for debugging in GitHub Actions)
print(f"Model Accuracy: {accuracy * 100}%")
