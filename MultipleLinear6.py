import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
dataset=pd.read_csv('/content/drive/My Drive/ML/california_housing.csv')
print(dataset.head())
features = dataset[['MedInc','HouseAge','AveRooms']]
response = dataset['MedHouseVal']
features_train,features_test,response_train,response_test=train_test_split(features,response,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(features_train,response_train)
predictions =model.predict(features_test)
mse =mean_squared_error(response_test,predictions)
r2 =r2_score(response_test,predictions)
print("Mean Squared Error;",mse)
print("R2 score:",r2)