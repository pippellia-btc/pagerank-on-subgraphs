# Pagerank on subgraphs—efficient Monte-Carlo estimation

In this repo you can find the reference code for my novel Monte-Carlo algorithm for efficiently computing the Pagerank distribution over S subgraph of G.
To play with it, follow the next steps

### Step 1: load the graph database

First, you have to download the networkx graph database into memory by running the following code

```python
import time
import json
import networkx as nx

# loading the database
print('loading the database...')
tic = time.time()

index_map, G = load_network(1714823396)

toc = time.time()
print(f'finished in {toc-tic} seconds\n')
```

### Step 2: Compute Pagerank over G

Compute the pagerank over G using the networkx built-in pagerank function that uses the power iteration method.
This vector will be considered as the real pagerank vector.

```python

# computing the pagerank
print('computing global pagerank...')
tic = time.time()

p_G = nx.pagerank(G, tol=1e-12)

toc = time.time()
print(f'finished in {toc-tic} seconds\n')
```

### Step 3: Approximate Pagerank over G using Monte-Carlo

Compute the pagerank over G using a simple Monte-Carlo implementation and compute the L1 error.
This step is essential because it gives us the csr-matrix `walk_visited_count`.

```python
R = 10

# computing the monte-carlo pagerank
print('computing monte-carlo pagerank')
tic = time.time()

walk_visited_count, mc_pagerank = get_mc_pagerank(G,R)

toc = time.time()
print(f'finished in {toc-tic} seconds')

# computing the L1 error
error_G_mc = sum( abs(p_G[node] - mc_pagerank[node]) for node in G.nodes())
print(f'L1 error = {error_G_mc}')
```

### Step 4: 
