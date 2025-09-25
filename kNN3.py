import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
file_path = '/content/drive/My Drive/ML/breast-cancer.csv'
df = pd.read_csv(file_path)
print(df.head())
features = df.drop(['id', 'diagnosis'], axis=1)
labels = df['diagnosis']
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(features_train, labels_train)
predictions = knn.predict(features_test)
accuracy = accuracy_score(labels_test, predictions)*100
confusion = confusion_matrix(labels_test, predictions)
report = classification_report(labels_test, predictions)

print("Confusion Matrix: \n", confusion)
print("\n\nAccuracy of the k-NN Classifier:", accuracy, "%")
print("\n\nReport: \n", report)