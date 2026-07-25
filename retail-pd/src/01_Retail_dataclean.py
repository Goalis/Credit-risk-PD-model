import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_load_retail import load_credit_data
import os

df=load_credit_data()

print(df)

resultat = pd.DataFrame({
    "mean": df.mean(numeric_only=True),
    "std": df.std(numeric_only=True),
    "missing": df.isna().sum()
})


text_cols = df.select_dtypes(include=["object", "string"]).columns

# Antal missing i tekstkolonner
missing_text = df[text_cols].isna().sum()
print(missing_text)

missing_text = missing_text[missing_text > 0]

print(missing_text)