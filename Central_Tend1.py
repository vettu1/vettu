import pandas as pd
import numpy as np
ds = pd.read_csv("/content/drive/MyDrive/ML/auto_mpg.csv")
print(ds)
ds.head()
ds.dropna(inplace=True)
mean_mpg = ds['mpg'].mean()
median_mpg = ds['mpg'].median()
mode_mpg = ds['mpg'].mode()[0]
print(f"mean_mpg: {mean_mpg}")
print(f"median_mpg: {median_mpg}")
print(f"mode_mpg: {mode_mpg}")
std_mpg = ds['mpg'].std()
variance_mpg = ds['mpg'].var()
print(f"standard Deviation_mpg: {std_mpg}")
print(f"variance_mpg: {variance_mpg}")
summary_stats = ds.describe()
print(summary_stats)