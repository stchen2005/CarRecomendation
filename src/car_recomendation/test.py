print("test")
print("test from desktop")

import pandas as pd

df = pd.read_csv("data/Cars Datasets 2025.csv", encoding="latin1")
print(df.head())
print(df.columns)
print(df.info())