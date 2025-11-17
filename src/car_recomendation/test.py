print("test")
print("test from desktop")

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

car_data = pd.read_csv("CarRecomendation\data\Cars Datasets 2025.csv", encoding="latin1")
# print(car_data.shape)
print(car_data.head())
print(car_data.dtypes)
label_encoder = LabelEncoder()
print(car_data.columns)
print(car_data.columns.size)
excluded = ['Cars Names', 'Company Names']
target = 'Performance(0 - 100 )KM/H'
for column in car_data.columns:
    if column not in excluded:
        car_data[column] = label_encoder.fit_transform(car_data[column].astype(str))
print(car_data.dtypes)

X = car_data.drop(excluded + ['Performance(0 - 100 )KM/H'], axis=1)
Y = car_data['Performance(0 - 100 )KM/H']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
categorical_features = ['Engines', 'Fuel Types']
numeric_features = ['CC/Battery Capacity', 'HorsePower', 'Total Speed', 'Cars Prices', 'Seats', 'Torque']

# Preprocessor: OneHot for categorical, StandardScaler for numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ],
    remainder='drop'  # drop excluded columns
)
#print(car_data.dtypes)
scaler = StandardScaler()
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
model = RandomForestRegressor(n_estimators=100, random_state=2)
model.fit(X_train_processed, Y_train)

# Evaluate
predictions = model.predict(X_test_processed)

print("MSE:", mean_squared_error(Y_test, predictions))
print("R²:", r2_score(Y_test, predictions))
new_car = {
    'Company Names': 'FERRARI',
    'Cars Names': '812 GTS',
    'Engines': 'V12',
    'CC/Battery Capacity': 6496,
    'HorsePower': 789,
    'Total Speed': 340,
    'Performance(0 - 100 )KM/H': 2.9,   # target column, usually not included in X
    'Cars Prices': 350000,
    'Fuel Types': 'Petrol',
    'Seats': 2,
    'Torque': 718
}
new_car_df = pd.DataFrame([new_car])
new_car_df = new_car_df.drop(columns=['Company Names', 'Cars Names', 'Performance(0 - 100 )KM/H'])
# Preprocess new car safely
X_new_processed = preprocessor.transform(new_car_df)
predicted_time = model.predict(X_new_processed)
print("Predicted 0-100 km/h time:", predicted_time[0])
print(new_car_df)
print(new_car_df.dtypes)
X_new_processed = preprocessor.transform(new_car_df)

print("Processed shape:", X_new_processed.shape)
print("Processed vector:", X_new_processed.toarray()[0])
X_train_processed_sample = preprocessor.transform(X_train.iloc[[0]])
print("Training sample vector:", X_train_processed_sample.toarray()[0])