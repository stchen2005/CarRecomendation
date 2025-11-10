print("test")
print("test from desktop")

import pandas as pd
import os
print("Current working directory:", os.getcwd())

df = pd.read_csv("CarRecomendation/data/Cars Datasets 2025.csv", encoding="latin1")

print(df.head())
print(df.columns)
print(df.info())