print("test")
print("test from desktop")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

car_data = pd.read_csv("data/Cars Datasets 2025.csv", encoding="latin1")
print(car_data.shape)
print(car_data['Seats'].value_counts)
encoder = LabelEncoder()
car_data['Seats'] = encoder.fit_transform(car_data['Seats'])

print(car_data.groupby("Seats").mean(numeric_only=True))
print(car_data.columns)
print(car_data.dtypes)