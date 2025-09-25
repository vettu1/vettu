import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv('/content/drive/MyDrive/ML/indian_liver_patient.csv')
print(dataset.head())
features = dataset.iloc[:, :-1]
target = dataset.iloc[:, -1]
categorical_cols = features.select_dtypes(include='object').columns.tolist()
print("\n\nCategorical columns:", categorical_cols)
if categorical_cols:
    features = pd.get_dummies(features, columns=categorical_cols, drop_first=True)
if features.empty or target.empty:
    raise ValueError("Features or target variable is empty.")
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(features_train, target_train)
predictions = model.predict(features_test)
accuracy = accuracy_score(target_test, predictions)
confusion_mat = confusion_matrix(target_test, predictions)
classification_rep = classification_report(target_test, predictions)
print("\n\nAccuracy:", accuracy)
print("\n\nConfusion Matrix:\n", confusion_mat)
print("\n\nClassification Report:\n", classification_rep)