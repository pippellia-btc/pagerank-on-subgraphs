# Pagerank on subgraphs—efficient Monte-Carlo estimation

In this repo you can find the reference code for my novel Subrank algorithm for efficiently computing the Pagerank distribution over $S$ subgraph of $G$.
To play with it, follow these steps:

## Step 1: load the graph database

First, you have to load the networkx graph database into memory by running the following code.

```python
import time

# loading the database
print('loading the database...')
tic = time.time()

index_map, G = load_network(1714823396)

toc = time.time()
print(f'finished in {toc-tic} seconds')
```

## Step 2: Compute Pagerank over $G$

Compute the pagerank over $G$ by using the networkx built-in pagerank function that uses the power iteration method.
This vector will be considered as the real Pagerank vector and will be used to compute the errors of the Monte-Carlo algorithm.

```python
import networkx as nx

# computing the pagerank
print('computing global pagerank...')
tic = time.time()

p_G = nx.pagerank(G, tol=1e-12)

toc = time.time()
print(f'finished in {toc-tic} seconds')
```

## Step 3: Approximate Pagerank over $G$ using Monte-Carlo

Compute the pagerank over $G$ using a simple Monte-Carlo implementation and compute the L1 error.
This step is essential because it returns the csr-matrix `walk_visited_count`, that will be used later by the Subrank algorithm.

```python
# number of the random walks per node
R = 10

# fix the order of the nodes
nodelist = list(G.nodes())

tic = time.time()

# perform the random walks and get the monte-carlo pagerank
walk_visited_count, mc_pagerank = get_mc_pagerank(G, R, nodelist)

toc = time.time()
print(f'performed random walks in {toc-tic} seconds')

# computing the L1 error
error_G_mc = sum( abs(p_G[node] - mc_pagerank[node]) for node in G.nodes() )
print(f'error pagerank vs mc pagerank in G = {error_G_mc}')
```

## Step 4: Select random subgraph $S$ and compute its Pagerank distribution

Select a random subgraph $S$ consisting of 50k nodes, and compute its Pagerank distribution.

```python
# selecting random subgraph S
S_nodes = set(random.sample(list(G.nodes()), k=50000))
S = G.subgraph(S_nodes).copy()

# computing pagerank over S
print('computing local pagerank...')
tic = time.time()

p_S = nx.pagerank(S, tol=1e-12)

toc = time.time()
print(f'finished in {toc-tic} seconds')
```

## Step 5: Approximate Pagerank over $S$ using Subrank

Run the Subrank algorithm to approximate the Pagerank over $S$ subgraph of $G$. Then compute the L1 error.

```python
# computing subrank
print('computing subrank over S...')
tic = time.time()

subrank = get_subrank(S, G, walk_visited_count, nodelist)

toc = time.time()
print(f'performed random walks in {toc-tic} seconds')

# computing the L1 error
error_S_subrank = sum( abs(p_S[node] - subrank[node]) for node in S_nodes )
print(f'error pagerank vs subrank in S = {error_S_subrank}')
```

## Step 6: Approximate Pagerank over $S$ using Monte-Carlo naive recomputation

Run the Monte-Carlo Pagerank algorithm on $S$ as a reference for the number of random walks required and the error achieved.

```python
# computing the monte-carlo pagerank 
print('computing naive monte-carlo pagerank over S')
tic = time.time()

_, mc_pagerank_S_naive = get_mc_pagerank(S,R)

toc = time.time()
print(f'finished in {toc-tic} seconds')

# computing the L1 error
error_S_naive = sum( abs(p_S[node] - mc_pagerank_S_naive[node]) for node in S.nodes())
print(f'error pagerank vs mc pagerank in S = {error_S_naive}')
```
