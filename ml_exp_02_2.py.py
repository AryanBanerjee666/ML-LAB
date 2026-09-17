import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)

plt.figure(figsize=(20,5))
sns.boxplot(data=df)
plt.title("Boxplots of Wine Dataset Features")
plt.show()
