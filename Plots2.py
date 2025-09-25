import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("/content/drive/MyDrive/ML/auto_mpg.csv")
df.head()
df.dropna(inplace=True)
sns.set(style="whitegrid")
plt.figure(figsize=(16, 10))
df.dropna(inplace=True)
sns.set(style="whitegrid")
plt.figure(figsize=(16, 10))

plt.subplot(2,3,1)
sns.histplot(df['mpg'], bins=20, kde=True)
plt.title('Distribution of MPG')
plt.xlabel('MPG')
plt.ylabel('Frequency')

plt.subplot(2,3,2)
sns.boxplot(x=df['mpg'])
plt.title('Boxplot of MPG')
plt.xlabel('MPG')

plt.subplot(2,3,3)
sns.scatterplot(x='weight', y='mpg', data=df)
plt.title('Scatterplot of Weight vs MPG')
plt.xlabel('Weight')
plt.ylabel('MPG')

plt.subplot(2,3,4)
sns.pairplot(df, diag_kind='kde', markers='o')
plt.suptitle('pair Plot of All numerical Features', y=1.02)