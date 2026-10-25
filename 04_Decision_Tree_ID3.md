# Decision Tree (ID3-style) Purchase Prediction

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('ecommerce.csv')  # Change filename
display(df.head())

# Features and target
target = 'Purchased' if 'Purchased' in df else df.columns[-1]  # Verify target
X = df.drop(columns=target)
X = X.loc[:, ~X.columns.str.lower().str.contains('id')]
y = df[target]
valid = y.notna(); X, y = X.loc[valid].copy(), y.loc[valid]

# Missing values
for c in X:
    if pd.api.types.is_numeric_dtype(X[c]): X[c] = X[c].fillna(X[c].median())
    else: X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else 'Unknown')

# Categorical encoding
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

# Decision Tree (entropy / information gain, ID3-style)
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Evaluation
print('Accuracy:', round(accuracy_score(y_test, pred) * 100, 2), '%')
print(classification_report(y_test, pred, zero_division=0))

# Visualizations
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()
plt.figure(figsize=(16, 7))
plot_tree(model, feature_names=X.columns, class_names=[str(c) for c in model.classes_], filled=True, fontsize=8)
plt.show()

# Sample prediction
print('Sample customer prediction:', model.predict(X_test.iloc[:1])[0])
```
