import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
diabetes_data = pd.read_csv('/content/drive/My Drive/ML/diabetes.csv')
print(diabetes_data.head())
features = diabetes_data.drop('Outcome',axis=1)
labels = diabetes_data['Outcome']
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.2,random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(features_train,labels_train)
predictions =model.predict(features_test)
accuracy = accuracy_score(labels_test,predictions)
confusion = confusion_matrix(labels_test,predictions)
report = classification_report(labels_test,predictions)
print("Accuracy of the classifier:",accuracy)
print("Confusion Matrix:\n",confusion)
print("Classification Report:\n",report)