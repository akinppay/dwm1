# Retail Discretization and Binning

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('retail.csv')  # Change filename
display(df.head())

# Select numerical features
num = [c for c in df.select_dtypes(include='number') if 'id' not in c.lower() and df[c].nunique() > 1]

# Missing values
df[num] = df[num].fillna(df[num].median())

# Descriptive analysis
display(df[num].describe())

# Discretization (equal-width binning)
for c in num:
    df[c + '_Bin'] = pd.cut(df[c], bins=3, labels=['Low', 'Medium', 'High'])
display(df.head())

# Visualizations
if num:
    df[num].hist(figsize=(12, 7)); plt.tight_layout(); plt.show()
    for c in num:
        df[c + '_Bin'].value_counts().reindex(['Low', 'Medium', 'High']).plot.bar(title=c + ' Bins')
        plt.ylabel('Customers'); plt.tight_layout(); plt.show()

# Insights
for c in num:
    print(c, 'distribution:\n', df[c + '_Bin'].value_counts().to_string())

# Save output
df.to_csv('retail_processed.csv', index=False)
```
