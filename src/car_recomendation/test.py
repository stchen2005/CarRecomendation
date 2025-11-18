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
from sklearn import svm
# -------------------------
# 1. Load dataset
# -------------------------
car_data = pd.read_csv("CarRecomendation/data/Cars Datasets 2025.csv", encoding="latin1")

# -------------------------
# 2. Clean numeric columns
# -------------------------
def clean_numeric(series):
    # Convert to string and remove commas and invalid chars
    cleaned = (
        series.astype(str)
        .str.replace(',', '', regex=False)
        .str.replace(r'[^0-9.]', '', regex=True)
    )

    # Fix values with multiple dots like "64.872.8"
    def fix_multiple_dots(value):
        if value.count('.') <= 1:
            return value
        parts = value.split('.')
        return ''.join(parts[:-1]) + '.' + parts[-1]

    cleaned = cleaned.apply(fix_multiple_dots)

    # Convert empty strings to NaN
    cleaned = cleaned.replace('', np.nan)

    # Convert to float
    return cleaned.astype(float)

numeric_cols = [
    'CC/Battery Capacity',
    'HorsePower',
    'Total Speed',
    'Cars Prices',
    'Torque',
    'Performance(0 - 100 )KM/H'
]

for col in numeric_cols:
    car_data[col] = clean_numeric(car_data[col])


# -------------------------
# 3. Clean Seats (handles "2+2")
# -------------------------
def clean_seats(value):
    value = str(value).strip()

    if "+" in value:
        try:
            return sum(int(x) for x in value.split("+"))
        except:
            return None

    if value.isdigit():
        return int(value)

    return None

car_data["Seats"] = car_data["Seats"].apply(clean_seats)


# -------------------------
# 4. Remove any remaining invalid rows
# -------------------------
car_data = car_data.dropna(subset=numeric_cols + ["Seats"])

# -------------------------
# 5. Define features AFTER cleaning
# -------------------------
categorical_features = ['Engines', 'Fuel Types']
numeric_features = ['CC/Battery Capacity', 'HorsePower', 'Total Speed',
                    'Cars Prices', 'Seats', 'Torque']

X = car_data.drop(columns=['Company Names', 'Cars Names', 'Performance(0 - 100 )KM/H'])
y = car_data['Performance(0 - 100 )KM/H']


# -------------------------
# 6. Preprocessor
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

# -------------------------
# 7. Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# -------------------------
# 8. Train model
# -------------------------
model = RandomForestRegressor(n_estimators=100, random_state=2)
model.fit(X_train_processed, y_train)

# -------------------------
# 9. Evaluate
# -------------------------
predictions = model.predict(X_test_processed)
print("MSE:", mean_squared_error(y_test, predictions))
print("R²:", r2_score(y_test, predictions))

# -------------------------
# 10. Predict a new car
# -------------------------
new_car = {
    'Engines': 'I4',
    'CC/Battery Capacity': 2300,
    'HorsePower': 315,
    'Total Speed': 180,
    'Cars Prices': 100,
    'Fuel Types': 'Petrol',
    'Seats': 4,
    'Torque': 350
}

new_car_df = pd.DataFrame([new_car])

X_new_processed = preprocessor.transform(new_car_df)
predicted_time = model.predict(X_new_processed)
print("Predicted 0–100 km/h time:", predicted_time[0])