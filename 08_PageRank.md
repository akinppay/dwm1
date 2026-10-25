# PageRank Web Page Ranking

```python
import pandas as pd
import networkx as nx

# Load dataset (first two columns: source page, destination page)
df = pd.read_csv('webpages.csv')  # Change filename

# Directed graph
G = nx.from_pandas_edgelist(df, source=df.columns[0], target=df.columns[1], create_using=nx.DiGraph())

# PageRank algorithm
rank = nx.pagerank(G, alpha=0.85)

# Display ranked pages
result = pd.DataFrame(rank.items(), columns=['Web Page', 'PageRank Score'])
result = result.sort_values('PageRank Score', ascending=False)
print('Web Page Rankings:')
display(result)
```
