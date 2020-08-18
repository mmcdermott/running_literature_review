# Research Artifacts Notes

In this document I'll summarize discrete research artifacts I've read, leaving notes.

Template
```
# Topic
## [PAPER_TITLE](<LINK>)
### Summary
### Notes Section 1
...
```

# Pre-training
## [GRAPH-BERT: Only Attention is Needed for Learning Graph Representations](https://arxiv.org/pdf/2001.05140.pdf)
### Summary
This paper both proposes (1) a new form of graph neural network based solely on attention links and (2) a method for graph based pre-training / fine-tuning based on (1) a local node-attribute task and (2) a structure recovery task (both of which feel somewhat analogous to the node identification and context-prediction of \[1\]). They focus specifically on solving two problems:
  1. Suspended Animation Problem (by same authors; borderline rejected from ICLR 2020) \[2\]
     This problem seems to be some nonsense. Their analysis is based on the assumption that fully-connected layers are linear, which would make repeated convolutions identical to a single convolutional step. This analysis allows them to realize a GCNN as a Markov Chain, from which they draw similar analysis to #2 but realizing some steady state cyclical patterns.
  2. Over-smoothing Problem (different authors, AAAI 2018) \[3\]
     This problem is that repeated application of the graph Laplacian smoothing operator (The typical "convolution" analog in GNNs) makes all vertices converge to the same value (per connected component) as edges act as smoothing operators.

### Model

They try to solve these two problems via their Graph-BERT model. This model takes the following form:
  1. Node Embeddings
     Nodes are embedded via several properties:
       - Raw embeddings
       - Weisfeiler-Lehman Absolute Role Embeddings, which are determined from the WL algorithm for determining a unique "role" for nodes in graphs. These roles are natural numbers perscribed per-node based on the full data, but (in this text) it isn't implied there is any natural distance in the WL-space. They use these WL roles via position embeddings (e.g., sins/cosines), which to me makes no sense given lack of any natural WL-distance.
       - Intimacy-based Relative Position Embeddings, which are Relative Position Representations based on a serialized view of the graph defined according to relative intimacy ranks of various pairwise relationships. This doesn't make too much sense to me - seems to be anchored too strongly to text; why not look for intra-graph distance metrics and use those?
       - Hop-based relative position representation.
  2. Attention over these embeddings. AFAICT, the only dependence here on graph structure is in the relative position representations.
  3. GNN layer
  4. Summary representation fusion (just average over the graph, in this case).

This is actually very similar to the architecture I proposed for Anna to study; shame we never got it to work.

### Pre-training & Fine-tuning Tasks

### References
\[1\] https://arxiv.org/abs/1905.12265

\[2\] https://arxiv.org/abs/1909.05729

\[3\] https://arxiv.org/abs/1801.07606

# Multi-task Learning
## [Which Tasks Should be Learned Together in Multi-task Learning?](https://arxiv.org/pdf/1905.07553.pdf)
### Summary
This paper investigates multi-task learning in computer vision, and proposes a scheme to identify what subset of tasks should be learned together. In particular, their key problem is:
> Given a set of tasks, T , and a computational budget b (e.g., maximum allowable inference time), what is the optimal way to assign tasks to networks with combined cost _≤ b_ such that a combined measure of task performances is maximized? - Page 2

### Task-Relationships Among Multi-task Learning
They find several notable findings:
  1. More tasks = worse performance in comparison to ST models at the same (individual) capacity level, but outperform ST models that are restricted to 1/N of the capacity budget (given N tasks)
  2. They link to another paper I should read: https://arxiv.org/abs/1804.08328 for a transfer task-specific mechanism.

# Neural ODEs
## [Neural Ordinary Differential Equations](https://arxiv.org/pdf/1806.07366.pdf)
### Summary
#### Pitch
Let's look at a residuatl neural network. This network realizes the output of layer `t` via `h_t(x) = h_{t-1}(x) + f(h_{t-1}(x); \theta)`. This looks like an Euler numerical solution to a differential equation: `h_x(t) = h_x(t-1) + \left.\frac{d}{dt}h_x \right|_{h_x(t-1)}`, where `f` is realized as the derivative of `h_x` (and thereby the derivative of `h_x` is realized as a single layer of a neural network (or a block of layers)).

This is sort of interesting, but why would we want to do this? 
  1. We can thus use an ODE solver to solve for `h_x(t)` _for any `t`_. This allows for
    a. Continuous depth ResNets.
    b. Continuous time RNNs.
    c. (as shown in this paper) constant memory solution for arbitrary depth of networks!
    d. (as shown in this paper) we can then solve for neural networks with more complex solvers!
    e. (as shown in this paper) this results in easier change-of-variables formula for pre-NN to post-NN systems, which yields easier use of normalizing flow networks.
#### Differentiating through an ODE Solver
TODO
#### Use cases
##### Continous Timeseries Models
In this use-case, the authors suppose the following probabilistic system:

![Continous TS probabilistic system](https://github.com/mmcdermott/running_literature_review/raw/master/research_artifacts/neural_ode_continuous_ts.png)

This system realizes the whole timeseries probabilistically, with an initial latent state distributed according to `p(z_0)`, and the latent state at subsequent timepoints governed by a neural-ODE model `z_{t_i} \sim ODESolve(z_0, f, \theta, t_i)`. An "emission" probability model `q(x | z)` enables the system to generate predictions in the raw data space.
