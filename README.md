# Pagerank on subgraphs—efficient Monte-Carlo estimation

In this repo you can find the reference code for my novel Subrank algorithm for efficiently computing the Pagerank distribution over $S$ subgraph of $G$.
For the reasoning behind the algorithm, the definition and the analysis, I invite the interested reader to [read the paper](https://pippellia.com/pippellia/Social+Graph/Pagerank+on+subgraphs%E2%80%94efficient+Monte-Carlo+estimation).

**Note**: The code is not intended for production use, as it lacks many performance optimizations, such as parallelization of random walks. Its purpose is to provide a reference implementation of the Subrank algorithm that is easy to read and understand.

To play with it, check out the Jupyter notebook `subrank_demo_notebook.ipynb`

## Files

| File  | Description |
| ------------- | ------------- |
| index_map_1714823396.json  | dictionary that maps Nostr public keys to nodes in the graph  |
| network_graph_1714823396.json  | Networkx graph of the Nostr network  |
| get_mc_pagerank.py | Implementation of the Pagerank algorithm Monte-Carlo complete path stopping at dandling nodes |
| subrank.py | Implementation of the Subrank algorithm for efficiently approximating the Pagerank on S subgraph of G |
| subrank_demo_notebook.ipynb | Jupyter notebook that shows a complete demo |

## Results

On random 3-hops subgraphs, the algorithm shows its best performance, reducing the number of random walks to be performed by ~94.6% on average. This is a considerable reduction in overhead compared to the naive algorithm that simply recomputes all random walks.

![number-of-walks-comparison](https://publish-01.obsidian.md/access/fd5a5849deab7856628935d9cba4ade8/Social%20Graph/Media/number-of-walks-comparison-naive-approx-algos-3hops.png)
![ratio-number-of-walks](https://publish-01.obsidian.md/access/fd5a5849deab7856628935d9cba4ade8/Social%20Graph/Media/ratio-number-of-walks-comparison-naive-approx-algos-3hops.png)

Despite the dramatic reduction in overhead, the algorithm achieves a very similar error compared to the naive recomputation.
![error-comparison](https://publish-01.obsidian.md/access/fd5a5849deab7856628935d9cba4ade8/Social%20Graph/Media/error-comparison-naive-approx-algos-3hops%201.png)

For the reasoning behind the algorithm, the definition and the analysis, I invite the interested reader to [read the paper](https://pippellia.com/pippellia/Social+Graph/Pagerank+on+subgraphs%E2%80%94efficient+Monte-Carlo+estimation).
