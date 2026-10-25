# KNN Disease Classification

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('diabetes.csv')  # Change filename
display(df.head())

# Features and target
target = 'Outcome' if 'Outcome' in df else df.columns[-1]  # Verify target
X = df.drop(columns=target).select_dtypes(include='number')
X = X.loc[:, ~X.columns.str.lower().str.contains('id')]
y = df[target]
valid = y.notna(); X, y = X.loc[valid], y.loc[valid]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42,
                                                    stratify=y if y.value_counts().min() >= 2 else None)

# Missing values and scaling
imputer, scaler = SimpleImputer(strategy='median'), StandardScaler()
X_train = scaler.fit_transform(imputer.fit_transform(X_train))
X_test = scaler.transform(imputer.transform(X_test))

# KNN classification
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# Evaluation
print('Accuracy:', round(accuracy_score(y_test, pred) * 100, 2), '%')
print(classification_report(y_test, pred, zero_division=0))

# Visualization
sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted'); plt.ylabel('Actual'); plt.show()

# Predict sample patient
print('Sample patient prediction:', model.predict(X_test[:1])[0])
```
