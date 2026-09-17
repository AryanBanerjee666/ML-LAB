import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

print("---First five rows---")
print( df.head())
print("\n---Dataset Information---")
print(df.info())
print("\n---Statistical Summary---")
print(df.describe())
print("\n---Missing Value---")
print(df.isnull().sum())

print("\n---Correlational Marrix---")
print(df.corr(numeric_only=True))

plt.figure(figsize=(7,5))
sns.scatterplot(
    data=df,
    x="alcohol",
    y="malic_acid",
    hue="target",
    palette="viridis"
)
plt.title("Alcohol vs Malic Acid")
plt.show()

plt.figure(figsize=(7,5))
sns.histplot(df["alcohol"],kde=True,color="green")
plt.title("Alcohol content")
plt.xlabel("Alcohol")
plt.ylabel("Frequency")
plt.show()