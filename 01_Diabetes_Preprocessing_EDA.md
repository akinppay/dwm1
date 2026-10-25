# Diabetes Preprocessing and EDA

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv('diabetes.csv')  # Change filename
display(df.head())

# Missing values
# In diabetes datasets, zero may indicate a missing measurement
for c in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    if c in df: df[c] = df[c].replace(0, np.nan)
df = df.replace([np.inf, -np.inf], np.nan)
num = df.select_dtypes(include='number').columns.tolist()
cat = df.select_dtypes(exclude='number').columns.tolist()
features = [c for c in num if c.lower() not in ['outcome', 'target', 'label', 'class'] and 'id' not in c.lower()]
df[features] = df[features].fillna(df[features].median())
for c in cat: df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else 'Unknown')
print('Missing values:\n', df.isna().sum())

# Descriptive statistics
display(df.describe(include='all'))

# Outlier treatment (IQR capping)
for c in features:
    q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
    df[c] = df[c].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

# Normalization and transformation
if features:
    df[features] = MinMaxScaler().fit_transform(df[features])
    df[features] = np.log1p(df[features])

# Visualizations
if features:
    df[features].hist(figsize=(12, 7)); plt.tight_layout(); plt.show()
    plt.figure(figsize=(9, 6))
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm'); plt.show()
    df[features].plot.box(figsize=(10, 4), rot=45); plt.tight_layout(); plt.show()

# Save output
df.to_csv('diabetes_processed.csv', index=False)
print('Experiment completed!')
```
