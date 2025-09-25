import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score,confusion_matrix,classification_report
from sklearn.datasets import load_breast_cancer
cancer_data = load_breast_cancer()
features = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
labels = pd.Series(cancer_data.target)
features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.2,random_state=42)
model =SVC(kernel='linear')
model.fit(features_train,labels_train)
predictions = model.predict(features_test)
accuracy = accuracy_score(labels_test,predictions)
confusion =confusion_matrix(labels_test,predictions)
report = classification_report(labels_test,predictions)
print("Accuracy of the SVM classifier:",accuracy)
print("Confusuion Matrix:\n",confusion)
print("Classification Report:\n",report)