import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
california_housing = fetch_california_housing()
data = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
data['MEDV'] = california_housing.target
print(data.head())
features = data[['MedInc','Population','Latitude']]
response = data['MEDV']
features_train, features_test, response_train, response_test = train_test_split(
    features, response, test_size=0.2, random_state=42
)
model = Ridge(alpha=1.0)
model.fit(features_train, response_train)
predictions = model.predict(features_test)
predicted_price = model.predict(
    pd.DataFrame([[6, 10, 18]], columns=['MedInc','Population','Latitude'])
)
mse = mean_squared_error(response_test, predictions)
print("Mean Squared Error:", mse)
print("Predicted price for 6-room house:", predicted_price[0])