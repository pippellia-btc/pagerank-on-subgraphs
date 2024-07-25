# Pagerank on subgraphs—efficient Monte-Carlo estimation

In this repo you can find the reference code for my novel Subrank algorithm for efficiently computing the Pagerank distribution over $S$ subgraph of $G$.
For the reasoning behind the algorithm, the definition and the analysis, I invite the interested reader to [read the paper](https://pippellia.com/pippellia/Social+Graph/Pagerank+on+subgraphs%E2%80%94efficient+Monte-Carlo+estimation).

Note: The code is not meant to be used in production, as it lacks many performance optimization such as the parallelization of random walks. Its purpose is to provide a reference implementation for the Subrank algorithm that is easy to read and understand.

To play with it, check out the Jupyter notebook `subrank_demo_notebook.ipynb`
