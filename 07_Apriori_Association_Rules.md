# Apriori Market Basket Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset (one transaction per row, items across columns)
df = pd.read_csv('supermarket.csv', header=None)  # Change filename; remove header=None if CSV has headers
df.columns = df.columns.astype(str)
display(df.head())

# Missing values and one-hot encoding
basket = pd.get_dummies(df.dropna(how='all').astype('string').fillna(''), prefix='', prefix_sep='')
basket = basket.T.groupby(level=0).max().T
basket = basket.drop(columns=[''], errors='ignore').astype(bool)

# Frequent itemsets
items = apriori(basket, min_support=.02, use_colnames=True)
print('Frequent itemsets:')
display(items.sort_values('support', ascending=False).head(10))

# Association rules
rules = association_rules(items, metric='confidence', min_threshold=.3) if not items.empty else pd.DataFrame()
print('Association rules:')
display(rules.sort_values('lift', ascending=False).head(10) if not rules.empty else rules)

# Visualization
basket.sum().sort_values(ascending=False).head(10).plot.bar(figsize=(10, 5), color='skyblue')
plt.title('Top 10 Purchased Items'); plt.ylabel('Frequency'); plt.xticks(rotation=45, ha='right')
plt.tight_layout(); plt.show()

# Save output
rules.to_csv('association_rules.csv', index=False)
print('Experiment completed!')
```
