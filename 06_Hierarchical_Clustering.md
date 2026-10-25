# Agglomerative Hierarchical Clustering

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv('biological.csv')  # Change filename
display(df.head())

# Sampling (prevents MemoryError on large datasets)
df = df.sample(n=min(500, len(df)), random_state=42).copy()

# Numerical features
features = df.select_dtypes(include='number')
features = features.loc[:, ~features.columns.str.lower().str.contains('id|cluster')]
features = features.loc[:, features.nunique() > 1]

# Missing values and scaling
X = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(features))

# Dendrogram
Z = linkage(X, method='ward')
plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=20)
plt.title('Hierarchical Clustering Dendrogram'); plt.ylabel('Distance'); plt.show()

# Agglomerative clustering
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
df['Cluster'] = model.fit_predict(X)

# Cluster analysis
print('Cluster counts:\n', df['Cluster'].value_counts().sort_index())
display(df.groupby('Cluster')[features.columns].mean())

# Visualization (PCA)
points = PCA(n_components=2).fit_transform(X)
sns.scatterplot(x=points[:, 0], y=points[:, 1], hue=df['Cluster'], palette='Set1', s=35)
plt.title('Biological Sample Clusters'); plt.xlabel('PC1'); plt.ylabel('PC2'); plt.show()

# Save output
df.to_csv('biological_clusters.csv', index=False)
```
