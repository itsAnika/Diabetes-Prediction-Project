import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
import joblib

print("Starting the training and evalution:\n")

# Data Loading

try:
    df = pd.read_csv('dataset/diabetes_prediction_dataset.csv')
    print("Dataset loaded successfully")
except FileNotFoundError:
    print("Error: Could not find 'diabetes_prediction_dataset.csv' in the 'dataset' folder.")
    exit()

# Data Processing
df = pd.get_dummies(df, columns = ['gender', 'smoking_history'], drop_first = True)

# Save the exact features column for the predict script
feature_columns = df.drop('diabetes',axis=1).columns.tolist()
joblib.dump(feature_columns, 'feature_columns.pkl')

# seperate features (X) and target (y)
X = df.drop('diabetes', axis=1)
y = df['diabetes']

# Data Splitting (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Data scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training

print("Training Random Forest model:\n")
model = RandomForestClassifier(n_estimators = 100, random_state=42)
model.fit(X_train_scaled, y_train)

# predictions for metrics
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Calculate Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)


# Print Diagonostic Report
print(" Model Diagonostic Report: ")
print("="*40)
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1-Score:  {f1 * 100:.2f}%")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("-" * 40)
print("Confusion Matrix:")
print(f"True Negatives (TN):  {conf_matrix[0][0]}  | False Positives (FP): {conf_matrix[0][1]}")
print(f"False Negatives (FN): {conf_matrix[1][0]}  | True Positives (TP):  {conf_matrix[1][1]}")
print("="*40)

joblib.dump(model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nSuccess! Artifacts have been saved")

