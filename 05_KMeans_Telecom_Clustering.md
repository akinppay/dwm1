# K-Means Telecom Customer Segmentation

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv('telecom.csv')  # Change filename
display(df.head())

# Numerical features (exclude IDs)
features = df.select_dtypes(include='number')
features = features.loc[:, ~features.columns.str.lower().str.contains('id|cluster')]
features = features.loc[:, features.nunique() > 1]

# Missing values and scaling
X = StandardScaler().fit_transform(SimpleImputer(strategy='median').fit_transform(features))

# K-Means clustering (3 clusters)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = model.fit_predict(X)

# Cluster analysis
print('Cluster counts:\n', df['Cluster'].value_counts().sort_index())
display(df.groupby('Cluster')[features.columns].mean())
if 1 < df['Cluster'].nunique() < len(df):
    print('Silhouette score:', round(silhouette_score(X, df['Cluster']), 3))

# Visualization (2D PCA)
pca = PCA(n_components=2)
points = pca.fit_transform(X)
centers = pca.transform(model.cluster_centers_)
plt.figure(figsize=(10, 6))
for i, color in enumerate(['red', 'blue', 'green']):
    mask = df['Cluster'] == i
    plt.scatter(points[mask, 0], points[mask, 1], s=12, alpha=.35, color=color, label=f'Cluster {i}')
plt.scatter(centers[:, 0], centers[:, 1], s=260, marker='X', color='yellow', edgecolor='black', label='Centroids')
plt.xlabel('PC1'); plt.ylabel('PC2'); plt.legend(); plt.show()

# Cluster behavior heatmap
means = df.groupby('Cluster')[features.columns].mean()
scaled = (means - features.mean()) / features.std().replace(0, 1)
sns.heatmap(scaled, annot=True, cmap='coolwarm'); plt.title('Cluster Feature Profiles'); plt.show()

# Save output
df.to_csv('telecom_clusters.csv', index=False)
```
