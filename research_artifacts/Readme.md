# Research Artifacts Notes

In this document I'll summarize discrete research artifacts I've read, leaving notes.

Template
```
# Topic
## [PAPER_TITLE](LINK)
### Summary
### Notes Section 1
...
```
# Machine Learning for Computational Biology
## [Deep Learning of High-Order Interactions for Protein Interface Prediction](https://dl.acm.org/doi/pdf/10.1145/3394486.3403110)
  > Liu Y, Yuan H, Cai L, Ji S. Deep Learning of High-Order Interactions for Protein Interface Prediction. InProceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining 2020 Aug 23 (pp. 679-687).
  
### Summary
This work analyzes the task of protein interface prediction. This task takes in two proetins, one called the ligand protein with amino acid sequences of length `N_l` and the other the receptor protein of length `N_r`, and predicts over the dense `N_l \times N_r` output space which amino acids will interact in the resulting protein complex. Existing approaches for this problem have either represented proteins as simple sequences, graphs, or 3D structures alone, and have not considered all representations in concert. Additionally, the key related work [12](http://papers.nips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf) on which they base their analyses treats the prediction of each possible interface point as an independent prediction, which ignores interactions between possible interface pairs (e.g., a medium affinity possible interface might serve as an interface in one protein complex, but in another the (locally) same interface candidate with a similar affinity may not serve as an interface b/c a higher-affinity option exists nearby which is used instead). This analogy is a bit off, because using a GNN to represent the nodes (as [12](http://papers.nips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf) does) actually can build into a node's representation knowledge of the other nodes available, so each prediction between amino acid `i` in the ligand and `j` in the receptor can actually have all the information about the whole ligand protein and receptor protein, but practically it is likely fair as this level of information aggregation is unlikely to emerge, I think.

### Methodology
This method first uses a graph model, using protein affinities as edges in the graph to produce rich node features. Next, a sequence model (in original amino acid order) produces a unified representation of all features.. Next, they use the sequential model outputs to construct a 3D representation where the dim-3 slice at dim-0 `i` and dim-1 `j` represents the combination of the ligand node features `i` and receptor node features `j`, combination either being just summation or concatenation. This 3D matrix is then processed via a 3D CNN model to a 2D matrix consisting of probabilities of an interface at a various `i,j` amino acid combination.

Their GNN approach seems relatively simple/standard. The immediate neighborhood around a node (1-hop) is summarized with node- and edge- specific matrices weighting the neighborhood, followed by (potentially weighted via softmax on a learned (global) vector `q`) averaging. 

One interesting note -- they also perform a form of data augmentation by simultaneously predicting internal interactions within a protein as well as interfaces across the proteins.

#### Training Strategy
This is a big model they're ultimately proposing to work with, so they employ a specialized training strategy breaking apart proteins into smaller sub-graphs

### Experiments
They work across 3 datasets, compare to a number of baselines, and tune hyperparameters with grid search.
  
### Key Take-aways
1. The weighted average approach appears to be slightly better than the raw averaging approach.
2. Long-distance relationships at the GNN level don't appear to be super important -- only 2 GNN layers are needed for optimal performance.
3. Augmenting with in-protein contact prediction is valuable.
4. As far as I can tell they don't offer commentary on the effect of their subsampled training paradigm.

### Questions / Comments
  1. It isn't clear to me why their featurization approach is inherently richer than one relying on 3D structures alone.
  2. Why use GNN + Sequence Model? Seems like you could just impose sequential order as another edge type in graph.
  3. Relatedly, is amino acid affinity the most effective kind of graph for protein representation? What about using this as a factor in informing attention in a transformer (thinking of that as an efficient fully connected GNN)?
  4. This seems inefficient. We anticipate protein interface prediction to be a sparse problem (e.g., most pairs don't interact). Is there anything we can leverage to increase efficiency given that expectation of sparsity? Are there any insights from image segmentation, or sparse transformers, for example?
  5. How does their subsampling procedure for proteins make sense given their claim that long-distance relationships are important? Is there any global aggregation? Does subsampled size matter?


# Pre-training
## [Variance-reduced Language Pretraining via a Mask Proposal Network](https://arxiv.org/abs/2008.05333v1)
### Summary
This work provides a theoretical validation for using an adversarial approach in a self-supervised masked language model style pre-training task. In particular, they show that if one examines the variance of the gradient of the MLM objective for BERT, this naturally decomposes into one term that corresponds to the members of the data batch (the "sentence variance") and one term that corresponds to the random choice of mask (the "mask variance"). 

### Variance Decomposition
The MLM objective is negative log likelihood of the correct word being re-identified; e.g.: `\sum_{i:x_i^M = \texttt{[MASK]}} - \log P(x_i | x^M, i ; \theta_{\text{enc}}`. The expected loss is this loss evaluated over the distribution of the training data:
```L(\theta_{\text{enc}} = \mathbb{E}_{x \sim P_X} \mathbb{E}_{x^M \sim \text{MaskDist}(x)}[\ell(\theta_\text{enc} ; x^M, x)]```
Note that this decomposition of the expectation is only possible as we assume the mask distribution `MaskDist` is _independent of the data distribution `P_X`_, an assumption that is _not_ true in any situation where the mask distribution is learned from the data somehow (at least, it doesn't if the loss changes as well). 

This latter point actually reveals my ignorance, and this paper's strength -- they don't frame their task as trying to learn a contrastive loss, but instead they frame it as _still learning the random mask approach_, but to do this they use a proposal distribution to minimize this _random mask_ loss most efficiently. Why would using a proposal distribution help learn this loss most efficiently? In essence, we want to estimate the _expectation_ of the random variable, and much of the domain of the RV doesn't contribute to this expectation. This is possibly a key issue in our early efforts with the protein adversarial random training, and may explain the need for us to include jointly a term that was strictly random. Let's get more technical about what they show:

For starters, let's repeat a theorem from Monte-Carlo sampling: 
```
Denote z_1, \ldots, z_T are iid sampled from the proposal distribution P_2. Then \frac{1}{T} \sum_{t} \frac{p_1(z_t)}{p_2(z_t)} f(z_t) is an unbiased estimator of \mathbb{E}_{z \sim P_1}[f(z)]. Var_{z \sim P_2} [\frac{p_1(z)}{p_2(z)} f(z)} is minimized if probability density function p_2(z) \propto \norm{f(z)}_2.
```
Why is this true? Well, its not a formal proof, but note that `p_2(z) \propto \norm{f(z)}_2 \implies p_2(z) = c \norm{f(z)}_2`, so then 
```Var_{z \sim P_2} [\frac{p_1(z)}{p_2(z)} f(z)} = Var_{z\sim P_2}[c p_1(z) \frac{f(z)}{\norm{f(z)}_2}]```
If (for simplicity, and as implied by our notation) we assume `f(z)` is scalar valued, then further:
```Var_{z\sim P_2}[c p_1(z) \frac{f(z)}{\norm{f(z)}_2}] = c^2 Var_{z \sim P_2}[p_1(z)]```
We can also likely further relate c to `f(z)` somehow (averaged or somehow otherwise aggregated over all `z` due to normalization constraints). This isn't, on its own, entirely illuminating for me. So, let's go to the source: [1](https://openreview.net/pdf?id=r8lrEqPpYF8wknpYt57j) (also listed below with separate commentary).

So, one issue here is that that statement of the theorem seems to be... wrong? Instead, you want `p_2(z) \propto p_1(z) \norm{f(z)}_2`. Note that this, in the immediate use case of this paper, doesn't change much, b/c `p_1(z)` is a uniform distribution, and therefore is a constant. Also, this paper really doesn't cite [1] enough, in my opinion.

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
The math here is pretty complicated, and, to be blunt, I don't understand it. Luckily, I don't really have to to use the method! This codebase exists to provide solvers in this space for PyTorch: [https://github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq).

That said, let's try to figure it out. So, we have our function `h_t(x)` which represents the `t`th layer of the (residual) neural network at input `x`. We can re-realize this function as `h_x(t)`, which outputs the same quantity, but now represents the solution of the neural network on input `x` at _continuous_ layer `t` (e.g., we could also ask for `h_x(\pi)`, whereas `h_{\pi}(x)` isn't defined). How to think about this is to realize the residual neural network as an ODE being solved via an Euler method, with `h_x(t) = h_x(t-1) + f(h_x(t-1), t-1, \theta)`, where `f` gives the novel layer block at position `t`, and represents `\frac{dh_x}{dt}` in our ODE framework. Now, normally, suppose we run the system for `T` layers. Then, we'd evaluate our loss `L` on `h_T(x)`. In our new framework, we can realize this as `L(h_x(T))`. But, as `h_x` is now a differentiable function, we can write `h_x(T) = h_T(x)` as `h_x(0) + \int_{0}^{T} \frac{dh_x}{dt} dt = h_x(0) + \int_0^T f(h_x(t), t, \theta)`. Thus, our loss is framed as: 
```L(h_x(T)) = L(h_x(0) + \int_0^T f(h_x(t), t, \theta))```
Our desire is to optimize `L` over `\theta`, across `x` drawn from some data distribution `p_X`. We therefore want to know how to compute `\frac{dL}{d\theta}`. Before going further, let's mark a slight generalization -- we can stop thinking about `0` as our definite first layer, and instead think about some more general `t_0`, and replace `T` with `t_1`.
```L(h_x(t_1)) = L(h_x(t_0) + \int_{t_0}^{t_1} f(h_x(t), t, \theta))```

Now, let's think about how to compute our desired derivative. By the chain rule, we can see:
```\frac{\partial L(h_x(t))}{\partial \theta} = \frac{\partial L}{\partial h_x} \frac{\partial h_x(T)}{\partial \theta}```
Following further (apologies for the lack of latex) we see
```\frac{\partial h_x(T)}{\partial \theta} = 0 (h_x(t_0) is presumed to be constant w.r.t \theta??) + \frac{\partial}{\partial \theta} \int_{t_0}^{t_1} f(h_x(t), t, \theta)```
Assuming some nicities (that I couldn't name offhand) we can switch differentiation and integration, and
```\frac{\partial h_x(T)}{\partial \theta} = \int_{t_0}^{t_1} \frac{\partial}{\partial \theta} f(h_x(t), t, \theta)```
Thus, we arrive at:
```\frac{\partial L(h_x(t))}{\partial \theta} = \frac{\partial L}{\partial h_x}|_{h_x(t_1)} \int_{t_0}^{t_1} \frac{\partial}{\partial \theta} f(h_x(t), t, \theta)```
... this looks... different than what is gotten in the ODEs paper.

##### Using existing resources!
Let's turn to existing resources to solve this discrepancy. We'll start with [this blog post](https://jontysinai.github.io/jekyll/update/2019/01/18/understanding-neural-odes.html)

One immediate discrepancy between my framing above and the blog post is that they realize `\theta` as _time-varying_, which I think likely makes sense. I'm not sure what effect that has on the math, though. They also clarify that they explicitly _assume_ that `h_x(t_0) = x`, which validates my assumption that `\frac{\partial h_x(t_0)}{\partial \theta} = 0`. Of course, if, instead, `h_x(t_0) = `*`z`*, where `z` is the initial latent space corresponding to `x`, then things may change (but I don't think they should, as the initial latent space doesn't depend on our parametrization.

Ok, here's the central discrepancy. At more than 3/4 of the way through, they say: "In the nueral ODE, our parameters are not just `\theta_t`, but _also the evaluation times `t_0` and `t_1`_" (emphasis mine). This may explain some of the additional derivatives the paper seems to calculate. Unfortunately, the rest of the blog post is useless for my purposes. However, in searching for confirmation of this in the paper itself, I'm led to the appendix, where more detail is exposed regarding the math. In particular, section B.1 presents a simple proof of the adjoint method I'd not understood, and B.2 explores the differentiation in more detail.

Ok, so B.2 makes things a bit clearer -- given the proof of the adjoint sensitivity method, we can use a clever trick to re-frame the ODE in an "augmented" space, expanded to include `\theta` and `t` directly. The adjoint method in this space (and its differential equation formulation) naturally given expressions then for the quantities of interest, _provided we make one key assumption_; namely, that `\frac{dL}{d\theta(t_N}} = 0`. Note that, in particular, this is presuming that the Neural network has converged. This smells like a connection to another of David Duvenaud's works: https://arxiv.org/abs/1911.02590

This work uses the implicit gradient method for hyperparameter optimization. This method _requires_ that the full algorithm be fully converged at optimization; this seems concordant with the use of Neural ODEs, where the process of solving directly enforces this assumption! Are there possibly other connections between these methods? Can they work synergestically together, somehow? What would this look like, exactly? The implicit function theorem (IFT) has been used to differentiate through arbitrarily large RNNs, so this makes me think there may be some connection here. 

#### Use cases
##### Continous Timeseries Models
In this use-case, the authors suppose the following probabilistic system:

![Continous TS probabilistic system](https://github.com/mmcdermott/running_literature_review/raw/master/research_artifacts/neural_ode_continuous_ts.png)

This system realizes the whole timeseries probabilistically, with an initial latent state distributed according to `p(z_0)`, and the latent state at subsequent timepoints governed by a neural-ODE model `z_{t_i} \sim ODESolve(z_0, f, \theta, t_i)`. An "emission" probability model `q(x | z)` enables the system to generate predictions in the raw data space.

#### Extensions
  1. Is it possible to combine this with https://arxiv.org/pdf/1911.02590.pdf somehow to enable better meta learning? What would that look like?
  2. How could we adapt this for graphs & other non-linear structures? Residual graph networks could maybe benefit from this, but I'd rather realize this through a more principled method, if possible?

### Follow-up work
This work has attracted a lot of follow-up work. What has that looked at?

### Additional Resources
  1. https://www.youtube.com/watch?v=YZ-_E7A3V2w

# Theory of Deep Learning
## Variance & SGD (may extend topic later)
### [Variance Reduction In SGD By Distributed Importance Sampling](https://openreview.net/pdf?id=r8lrEqPpYF8wknpYt57j)
This paper explores the genral idea of importance sampling SGD - an approach to reduce variance, principally explored in the context of asynchronous SGD (e.g., SGD distributed across multiple machines so small updates may be made on slightly stale gradient estimates as lag is induced via the distributed nature of the computation).

#### Theorem
Starting at the following theorem, as this paper is included to help better understand a different piece, they claim that if you are trying to estimate `E_{p(x)}[f(x)]`, you can minimize the variance of your estimator by instead viewing the system as `E_{q(x)}[\frac{p(x)}{q(x)} f(x)]` where `p(x) > 0 \implies q(x) > 0` and `q(x)` (your proposal distribution) is chosen such that `q(x)\propto p(x) |f(x)|`. 

They propose a scheme where they are able to implement this exactly and profile it. They find some interesting findings, concordant with their theory, and the (slightly more modern) understanding that some variance is helpful in SGD for regularization. Their immediate method is not applicable with most useful network types, unfortunately.

#### Follow-up work
This seems like an important technique, particularly for enabling pre-training algorithms to be run on much more limited computational resources, given the primary cost there is the decrease to batch size, which has a primary effect of increasing gradient variance. Some notable follow-up works that may warrant additional attention include:
  * https://jmlr.org/papers/volume19/16-241/16-241.pdf
  * (thesis): http://www.cs.cmu.edu/~weiyu/Adams_Wei_Yu_Homepage_files/thesis/thesis.pdf
