# Research Paper Quick Hits
This doc contains brief notes on _skimmed_ papers (no more than 20 min) per paper, which ultimately becomes a feeder into the more in depth notes in `README.md`. Papers are loosely organized into project/topic buckets.

### Acronyms
  1. WDTMT?: "What does this mean, technically?"
  2. ITT?: "Is this true?"
  
### Expected Format
#### Summary v1 (~ 15 min)
Goal is to decide if should be read or not in depth.
```
## [Paper_Title](paper_link)
  * **Logistics**:
    - Citation (key points: publication venue, date, authors)
    - Time Range: DATE (START - END)
  * **Summary**:
    - _Single Big Problem/Question_ The single big problem the paper seeks to solve (1 sent).
    - _Solution Proposed/Answer Found_ The proposed solution (1 sent).
    - _Experiments used to justify?_ 1-2 sentences or list on experiments
  * **Key Questions**:
    - Key questions (such that answers are necessary to decide if should continue or not).
  * **Key Strengths**:
    - List of big pros of the paper
  * **Key Weaknesses**:
    - List of big cons of this paper
  * **Warrants further read**: Y/N
```

#### Summary v2 (~ 30 min)
Goal is to give a full, complete summary of the paper.
```
## [Paper_Title](paper_link)
  * **Logistics**:
    - Citation (key points: publication venue, date, authors)
    - \# of citations of this work & as-of date
    - Author institutions
    - Time Range: DATE (START - END)
  * **Summary**:
    - _Single Big Problem/Question_ The single big problem the paper seeks to solve (1 sent).
    - _Solution Proposed/Answer Found_ The proposed solution (1 sent).
    - _Why hasn't this been done before?_ Why has nobody solved this problem in this way before? What hole in the literature does this paper fill? (1 sent).
    - _Experiments used to justify?_
      1) List of experiments used to justify (tasks, data, etc.) -- full context.
    - _Secret Terrible Thing_ What is the "secret terrible thing" of this paper?
    - 3 most relevant other papers:
      1)
      2)
      3)
    - Warrants deeper dive in main doc? (options: No, Not at present, Maybe, At least partially, Yes)
  * **Detailed Methodology**:
    Detailed description of underlying methodology
  * **Key Strengths**:
    - List of big pros of the paper
  * **Key Weaknesses**:
    - List of big cons of this paper
  * **Open Questions**:
    - List of open questions inspired by this paper
  * **Extensions**:
    - List of possible extensions to this paper, at any level of thought-out.
  * **How to learn more**:
    - List of terms/concepts/questions to investigate to learn more about this paper.
```
# Uncategorized
## [The Lottery Ticket Hypothesis for Pre-trained BERT Networks](https://arxiv.org/pdf/2007.12223.pdf)
  * **Logistics**:
    - Chen T, Frankle J, Chang S, Liu S, Zhang Y, Wang Z, Carbin M. The Lottery Ticket Hypothesis for Pre-trained BERT Networks. arXiv preprint arXiv:2007.12223. 2020 Jul 23.
    - Texas A&M, MIT CSAIL
    - Time Range: 10/2/20 (16:14 - 16:27 (stopped due to time))
  * **Summary**:
    - _Single Big Problem/Question_ Does BERT follow the lottery ticket hypothesis?
    - _Solution Proposed/Answer Found_ Yes, sort-of. Winning tickets are found in BERT systems, and beyond this subnetworks found at pre-training time _transfer_ to subnetworks for downstream tasks. However, something new is found, which is that subnetworks for FT tasks can be detected at initialization, unlike in general domain.
    - _Why hasn't this been done before?_ 
    - _Experiments used to justify?_ 
    - _Secret Terrible Thing_ 
    - 3 most relevant other papers:
      1) 
      2) 
      3)
    - Warrants deeper dive in main doc? (options: No, Not at present, Maybe, At least partially, Yes)
  * **Detailed Methodology**:
  * **Key Strengths**:
    - 
  * **Key Weaknesses**:
    - 
  * **Open Questions**:
    - 
  * **Extensions**:
    - 
  * **How to learn more**:
    - 

## [Rigging the Lottery: Making All Tickets Winners](https://arxiv.org/pdf/1911.11134.pdf)
  * **Logistics**:
    - Evci U, Gale T, Menick J, Castro PS, Elsen E. Rigging the lottery: Making all tickets winners. arXiv preprint arXiv:1911.11134. 2019 Nov 25.
    - 16 by 10/2/2020
    - Google Brain, Deep Mind
    - Time Range: 10/2/20 (15:38 - 16:11)
  * **Summary**:
    - _Single Big Problem/Question_ We want to train sparse neural networks from scratch that acheive comparable results as dense neural networks (e.g., just train the winning lottery tickets).
    - _Solution Proposed/Answer Found_ 
    - _Why hasn't this been done before?_ Lottery ticket is relatively recent (2018).
    - _Experiments used to justify?_ An extensive set of benchmarks used to validate RigL.
      1) TODO
    - _Secret Terrible Thing_ 
    - 3 most relevant other papers:
      1) Lottery Ticket
      2) Unstructured sparsity: https://myrtle.ai/wp-content/uploads/2019/06/IEEEformatMyrtle.ai_.21.06.19_b.pdf, https://arxiv.org/pdf/2008.11849.pdf
      3)
    - Warrants deeper dive in main doc? (options: No, Not at present, Maybe, At least partially, Yes)
  * **Detailed Methodology**:
    RigL uses the following algorithm: 
      1) Start with random sparsity. Train a while.
      2) Randomly update sparsity by eliminating edges based on magnitude, adding in edges based on instantaneous gradient information (WDTMT?)
      3) Repeat
    The really interesting part is how they _add_ edges. This requires computing gradients of the sparse parameters, which they somehow do in a manner that doesn't yield tons of increased computation! Turns out they don't do this cleverly. They comute them in a dense manner, then discard them promptly. If they can't store them all, then they compute them in an online manner and only store the top-k.
  * **Key Strengths**:
    - Strong evaluation & results
  * **Key Weaknesses**:
    - Unstructured sparsity may not buy you as much
    - Still need to compute dense gradients
  * **Open Questions**:
    - Would this work for BERT? In PyTorch?
    - Most interesting -- there are no lottery tickets in RigL. In RigL, all tickets seem to win. Why?
  * **Extensions**:
    - Can you leverage gradients of higher layers to inform which gradients need be computed at lower form? This might be able to be done by just using low-resolution (e.g. 16-bit or 8-bit precision) for early gradient calculation, so things fall to zero.
    - Can this be compbined with adaptive [unstructured sparsity](https://arxiv.org/pdf/2008.11849.pdf) computation?
  * **How to learn more**:
 
## [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
](https://arxiv.org/abs/1803.03635)
  * **Logistics**:
    - Frankle J, Carbin M. The lottery ticket hypothesis: Finding sparse, trainable neural networks. arXiv preprint arXiv:1803.03635. 2018 Mar 9.
    - 175 (as of 10/2/20)
    - MIT CSAIL
    - Time Range: 10/2/20 (13:35 - 13:58)
  * **Summary**:
    - _Single Big Problem/Question_ Question: Why do neural networks work and can we (and if so how can we) find small subnetworks which we can train from the start to achieve comparable performance?
    - _Solution Proposed/Answer Found_ The authors discover that the small, subnetworks identified via NN pruning methods (which traditionally obtain nearly the same performance as the original network) actually _can_ be trained successfully to acheive the same performance numbers. In addition they build on this to propose _iterative pruning_, in which they partially train a network, prune to the "winning ticket so far", then zero everything else out and continue training, until they perform another round of iterative pruning.
    - _Why hasn't this been done before?_ Unclear. May be one of those super rare examples of "nobody thought to check that before".
    - _Experiments used to justify?_
      1) Experiments on MNIST and CIFAR
    - _Secret Terrible Thing_ This requires warmup for deeper networks at larger learning rates. More complex networks require more specialized pruning strategies.
    - 3 most relevant other papers:
      1) Canonical pruning papers
      2) Any connections to optimization literature
      3) Follow up works:
    - Warrants deeper dive in main doc? Yes, eventually.
  * **Detailed Methodology**:
    It's actually really simple: Train for a time, then set the lowest n% of weights to 0, then iterate.
  * **Key Strengths**:
    - Impressive results with simple setup
    - Lots of theoretical implications
    - Winning tickets also generalize better.
  * **Key Weaknesses**:
    - Figures aren't great.
  * **Open Questions**:
    - Lots... Why is this how NNs work, does this apply to all architectures, can we improve performance or generalizability by finding more than 1 winning ticket (perhaps by re-randomly initializing weights rather than setting to 0 during iterative pruning), do our training regimes prioritize one ticket over several, what does this imply about learning, optimization, or generalization?
  * **Extensions**:
    - Randomly re-initialize weights of non winning ticket to generate more winners?
    - Can we predict the winning ticket in advance given network structure at initialization? 
    - Can we generate winning initializations?
    - Can we partially prune a network?
    - My thinking is that the winning ticket starts in a region of the optimization landscape that has, by chance, a convex path to a strong local optima. If this is true, we should be able to design an objective function that doesn't permit the system to learn winning tickets. Can we do this?
  * **How to learn more**:
    - Read more on pruning & follow-on work.


## [Imbalanced Image Classification with Complement Cross Entropy](https://arxiv.org/pdf/2009.02189v1.pdf)
  * **Logistics**:
    - Kim Y, Lee Y, Jeon M. Imbalanced Image Classification with Complement Cross Entropy. arXiv preprint arXiv:2009.02189. 2020 Sep 4.
    - Time Range: 09/24/2020 (12:35pm - 12:46pm)
  * **Summary**:
    - _Single Big Problem/Question_ Learning on imbalanced data is hard b/c cross-entropy loss mostly ignores output scores on wrong class (ITT?). A strategy proposed to address this is the use of complement cross entropy, but the proposed existing strategy is inefficient. This work makes it more efficient.
    - _Solution Proposed/Answer Found_ Uses "Complement Cross Entropy (CCE)" which sums the standard cross-entropy of prediction and the complement cross entropy (weighted by a balancing factor that normalizes scale), which is the mean of the sample-wise entropy on incorrect classes per each single batch. Note they don't propose this -- it is proposed first in [8](https://arxiv.org/abs/1903.01182). Instead, they just make it more efficient, summing losses instead of doing a 2-step procedure.
    - _Experiments used to justify?_ CIFAR-10 & 100 (with artificially induced class imbalance), Road Marking Dataset. They compare to ERM, COT [8](https://arxiv.org/abs/1903.01182), and focal loss \[24]. 
  * **Key Questions**:
    1. What is CCE, mathematically? What assumptions does it reflect / probabilistic quantities does optimizing for it optimize?
  * **Key Strengths**:
    - Does better than COT (maybe, no variance reported).
  * **Key Weaknesses**:
    - Doesn't introduce Complement cross entropy -- may thus not be worth reading primarily.
    - No Theory
  * **Warrants further read**: N, but should check out [8](https://arxiv.org/pdf/1903.01182.pdf)
  
## [COMPLEMENT OBJECTIVE TRAINING](https://arxiv.org/pdf/1903.01182.pdf)
  * **Logistics**:
    - Chen HY, Wang PH, Liu CH, Chang SC, Pan JY, Chen YT, Wei W, Juan DC. Complement Objective Training. InInternational Conference on Learning Representations 2018 Sep 27.
    - Time Range: 09/24/2020 (12:48pm - 12:55pm)
  * **Summary**:
    - _Single Big Problem/Question_ Learning on imbalanced data is hard b/c cross-entropy loss mostly ignores output scores on wrong class (ITT?). A strategy proposed to address this is the use of complement cross entropy
    - _Solution Proposed/Answer Found_ Uses "Complement Objective Entropy (COT)" which alternates between optimizing the standard cross-entropy of prediction and the complement cross entropy, which is the mean of the sample-wise entropy on incorrect classes per each single batch. Optimizing this loss encourages the model to yield uniform (maximal entropy) predictions across incorrect samples, which authors postulate improves generalizability by ensuring a larger gap between correct + incorrect examples.
    - _Experiments used to justify?_ A bunch. 
  * **Key Questions**:
    1. Does this make model more vulnerable to adversarial examples? _No, in contrast, it's apparently *more* robust!_
    2. Why not just sum losses? Unclear -- that is what is done in paper above.
  * **Key Strengths**:
    - Neat approach to improve classification performance, especially in classes of class imbalance.
  * **Key Weaknesses**:
    - No real theory -- why does this help, what does it optimize, probabilistically? Is it convergence speed, or true improved optimality? Does it lead to biases in predicted probabilities for cases with true uncertainty?
  * **Warrants further read**: No, it isn't relevant to my work.
  
## [Too Much Information Kills Information: A Clustering Perspective](https://arxiv.org/pdf/2009.07417v1.pdf)
  * **Logistics**:
    - Xu Y, Chau V, Wu C, Zhang Y, Zissimopoulos V, Zou Y. Too Much Information Kills Information: A Clustering Perspective. arXiv preprint arXiv:2009.07417. 2020 Sep 16.
    - Time Range: 09/24/2020 (12:56pm - 1:10pm)
  * **Summary**:
    - _Single Big Problem/Question_ Clustering is important, but is often (in particular, for variance-based `k`-clustering, which seeks to find a `k` sized partition of a dataset so as to minimize the sum of intra-cluster variances) computationally intensive, requiring careful initialization and potentially many passes through the entire dataset. Additionally, many existing clustering algorithms don't produce _balanced clusters_, where clusters must obey size constraints, is itself at computationally hard.
    - _Solution Proposed/Answer Found_ A new algorithm based on random sampling that yields provably good `k`-clustering results or, under an extension, balanced `k`-clustering tasks with hard constraints. This method not only yields strong clusters, but does so with less data, requiring only 7% of the data to yield clusters competitive with `k`-means and `k`-means++. This method works by taking a small random sample of the overall datasets, generating all possible k-clusters in this random subset, using the centroids induced by this sset to define clusters in the full dataset, and picking the best possible clustering out of those.
    - _Experiments used to justify?_  They use both theory and some numerical analyses on several datasets, both synthetic (generation process unspecified) and the real-world Cloud dataset.
  * **Key Strengths**:
    - Nice theoretical analysis showing that sampling based algorithm works well here.
  * **Key Weaknesses**:
    - Not sure this evaluation is fair (depends on whether they use train/test split), whether this scaling makes sense -- how much will you need on real datasets for this to matter? -- and whether or not this is any different than just a true random cluster search. Not sure they compare to the right baselines too (e.g., many runs of k-means on their random subset then pick the best one).
  * **Warrants further read**: N - isn't relevant to my research at present and not sure it is sufficiently theoretically of interest.

# Structured Biomedcial Pre-training
## [Contrastive Multi-View Representation Learning on Graphs](https://proceedings.icml.cc/static/paper_files/icml/2020/1971-Paper.pdf)
  * **Logistics**:
    - Hassani K, Khasahmadi AH. Contrastive Multi-View Representation Learning on Graphs. arXiv preprint arXiv:2006.05582. 2020 Jun 10.
    - Cited by 3
    - Time Range: 10/1/20 (13:49 - 13:57)
  * **Summary**:
    - _Single Big Problem/Question_ Node & graph level pre-training via contrastive learning, and asking whether findings in traditional multi-view representation learning generalize to graphs.
    - _Solution Proposed/Answer Found_ They use a diffusion algorithm to generate "another view" of a graph (looks like this just means adding or changing edges?) of a graph, then extracts subgraphs and embeds them, pushing the two subgraphs to be close to one another with a contrastive loss to push them to be apart from other, non-diffused graphs.
    - _Experiments used to justify?_ They examine 3 node classification tasks and 5 graph classification tasks, and compare to a large number of baselines.
  * **Key Questions**:
    - How does their objective really work? Does "diffusing" a graph in this way really make sense? What structural assumptions does it make?
  * **Key Strengths**:
    - Use both node-classification and graph classification.
  * **Key Weaknesses**:
    - Unsure about their diffusion estimation.
  * **Warrants further read**: Y, to answer key question, and only if graph pre-training / unsupervised graph learning becomes relevant.

## [CAGNN: Cluster-Aware Graph Neural Networks for Unsupervised Graph Representation Learning](https://arxiv.org/pdf/2009.01674.pdf)
  * **Logistics**:
    - Zhu Y, Xu Y, Yu F, Wu S, Wang L. CAGNN: Cluster-Aware Graph Neural Networks for Unsupervised Graph Representation Learning. arXiv preprint arXiv:2009.01674. 2020 Sep 3.
    - Time Range: 10/1/20 (13:36 - 13:44)
  * **Summary**:
    - _Single Big Problem/Question_ Unsupervised graph representation learning 
    - _Solution Proposed/Answer Found_ Zhu et al. use a clustering over the node embeddings as a self-supervised prediction objective for a GNN model, which is further augmented by strengthing intra-cluster edges and culling inter-cluster edges.
    - _Experiments used to justify?_ Zhu et al. use their method for graph pre-training, then perform node classification over Cora, Citeseer, and Pubmed citation networks, predicting article subject categories, using sparse BOW features for nodes, along with analyzing a node clustering task, with node classification labels being used to gauge cluster quality.
  * **Key Strengths**:
    - Reasonable experiments and nice results.
  * **Key Weaknesses**:
    - This seems to assume a particular kind of graph topology; namely, a  similarity graph structure. Graphs where connectivity patterns, rather than shortest-path distance, denote similarity, would likely fare poorly here.
    - They don't compare to GPT-GNN.
    - I'm concerned about degernative solutions to their task, though they try to solve this problem by requiring clusters to b equipartitioned and use an overclustering strategy (which has been previously explored).
  * **Warrants further read**: Maybe

## [GPT-GNN: Generative Pre-Training of Graph Neural Networks](https://dl.acm.org/doi/abs/10.1145/3394486.3403237?casa_token=S0J-HgsB_S8AAAAA:hmhyd5r-zWUvnsEV9GMfwWJbmVDGHslhCi-XPNBQ-eYnWRGIJSsCdMSHAmQupt0vbYEkDzGEQrmJ9g)
  * **Logistics**:
    - Hu Z, Dong Y, Wang K, Chang KW, Sun Y. GPT-GNN: Generative Pre-Training of Graph Neural Networks. InProceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining 2020 Aug 23 (pp. 1857-1867).
    - Cited by 1
    - Time Range: 10/1/20 (13:24 - 13:34)
  * **Summary**:
    - _Single Big Problem/Question_ Pre-training over graphs for node-classification, link prediction, or subgraph classification (though they only empirically analyze node classification).
    - _Solution Proposed/Answer Found_ Hu et al propose a generative pre-training objective based on masked graph generation, masking both nodes and edges.
    - _Experiments used to justify?_ Comparison across the open academic grpah and the amazon graph for node classification, comparing to graph autoencoder, graph sage, and graph infomax. They also show that the HGT offers strong performance as the base model here.
  * **Key Strengths**:
    - Strong experimental results, reasonable idea.
  * **Key Weaknesses**:
    - Only analyzes node classification, in practice.
  * **Warrants further read**: Y, and in addition the [Heterogenous Graph Transformer](https://dl.acm.org/doi/abs/10.1145/3366423.3380027?casa_token=neLIzbBgs70AAAAA:9PVQ1y_5p06rxyI28--hR5D6dGFH2e9_FBEJoxh_SxwJCYMOCiRHjil_lU8tyFDY3klfyx15OUuzsA) paper also warrants a look.


## [Neuro-symbolic representation learning on biological knowledge graphs](https://academic.oup.com/bioinformatics/article/33/17/2723/3760100)
  * **Logistics**:
    - Alshahrani M, Khan MA, Maddouri O, Kinjo AR, Queralt-Rosinach N, Hoehndorf R. Neuro-symbolic representation learning on biological knowledge graphs. Bioinformatics. 2017 Sep 1;33(17):2723-30.
    - Cited by 67
    - Time Range: 9/29/20 (13:51 - 13:55)
  * **Summary**:
    - _Single Big Problem/Question_ Featurize biomedical KGs.
    - _Solution Proposed/Answer Found_ Combine symbolig logic & NN approaches/
    - _Experiments used to justify?_ 
  * **Key Questions**:
    - How do they integrate KG graph-level features and node-level features, if at all? They don't
  * **Key Strengths**: (skipped)
  * **Key Weaknesses**: (skipped)
  * **Warrants further read**: No; it only applies graph level reasoning.


## [Annotating Gene Ontology terms for protein sequences with the Transformer model
](https://www.biorxiv.org/content/10.1101/2020.01.31.929604v1.abstract)
  * **Logistics**:
    - Duong DB, Gai L, Uppunda A, Le D, Eskin E, Li JJ, Chang KW. Annotating Gene Ontology terms for protein sequences with the Transformer model. bioRxiv. 2020 Jan 1.
    - Time Range: 9/29/20 (13:43 - 13:51)
  * **Summary**:
    - _Single Big Problem/Question_ Predict GO terms given a sequence, potentially augmented with additional data.
    - _Solution Proposed/Answer Found_ Transformers outperform CNNs.
    - _Experiments used to justify?_ Comparsion with DeepGO
  * **Key Questions**:
    - How do they encode PPI interactions? Using [Neuro-symbolic representation learning on biological knowledge graphs](https://academic.oup.com/bioinformatics/article/33/17/2723/3760100).
  * **Key Strengths**:
    - Reasonable comparison
    - Nice model
    - Nice ablations
  * **Key Weaknesses**:
    - No methodological novelty
  * **Warrants further read**: No, but one of its references does: [Neuro-symbolic representation learning on biological knowledge graphs](https://academic.oup.com/bioinformatics/article/33/17/2723/3760100).


## [Multifaceted protein–protein interaction prediction based on Siamese residual RCNN](https://academic.oup.com/bioinformatics/article/35/14/i305/5529260)
  * **Logistics**:
    - Chen M, Ju CJ, Zhou G, Chen X, Zhang T, Chang KW, Zaniolo C, Wang W. Multifaceted protein–protein interaction prediction based on siamese residual rcnn. Bioinformatics. 2019 Jul 15;35(14):i305-14.
    - 28 citations as of 09/28/20
    - UCLA
    - Time Range: 09/28/20 (main paper: 15:09 - 15:36, extending papers: 15:36 - )
  * **Summary**:
    - _Single Big Problem/Question_ Sequence-based protein-protein interaction (PPI) prediction is challenging, especially the more challenging sub-problems of interaction type prediction and binding affinity estimation.
    - _Solution Proposed/Answer Found_ Chen et al. propose the PPI Prediction based on a Twin Residual RCNN (PIPR) model, which use a twin residual CNN model to encode protein sequences and predict protein interaction.
    - _Why hasn't this been done before?_ PPI prediction has historically been done using hand-crafted expert features. This is just one more instance of ML being integrated more and more closely into the biomedical sciences.
    - _Experiments used to justify?_
      1) Guo's datasets: A collection of balanced datasets from various species for binary prediction of PPI.
      2) STRING datasets: Randomly sampled proteins from STRING (subject to sequence identity constraints) with multi-class PPI labels.
      3) SKEMPI dataset: Protein binding affinity data from SKEMPI. 3047 binding affinity (K_d) changes upon mutation within a protein complex.
      
      In addition, they also compare to a large number of baselines.
    - _Secret Terrible Thing_ Unclear
    - 3 most relevant other papers:
      1) Dataset Papers
      2) Alternative, graph-centric approaches, that may be multi-modal in nature:
        a) [Bio-JOE](https://www.biorxiv.org/content/10.1101/2020.06.15.153692v1.full.pdf) This paper produces a multi-modal KG embedding of biological networks for predicting interactions between COVID and human proteins. No sequence modelling component, but still likely warrants a citation in our paper.
        b) [This work](https://www.sciencedirect.com/science/article/pii/S2162253120302547) uses sequences and embedded GO terms simultaneously, but their overlap with us is minimal.
      3) Citing (downstream) papers:
        a) [MuPIPR](https://academic.oup.com/nargab/article/2/2/lqaa015/5781175): Very much like PIPR, but using an ELMO inspired contextual language-model representation of the amino acid seq rather than word2vec inspired, and focused solely on SKEMPI. Unclear if they actually beat PIPR, though.
        b) [Transforming the Language of Life](https://www.biorxiv.org/content/10.1101/2020.06.15.153643v1.full.pdf) uses a transformer PRoBERTa model to do PPI detection, on their homebrewed human Binary PPI "HIPPIE" dataset. On this dataset, they beat all baselines handily (by a suspicious margin).
        c) [Conjoint Feature Representation of Gene Ontology and...](https://www.sciencedirect.com/science/article/pii/S2162253120302547) this paper uses sequence and GO terms to do PPI prediction, achieving best in class performance, albeit leveraging additional data you're unlikely to have about proteins which are truly novel.
    - Warrants deeper dive in main doc? Yes, if used in downstream project.
  * **Detailed Methodology**:
    - Both proteins passed through same (twin) Res Recurrent CNN (Res RCNN), then element-wise multiplied, and fed through a ff net for prediction. 
    - Raw amino acid sequences are used, featurized via (1) pre-trained amino acid embeddings (word2vec style, not pre-trained style) and (2) one-hot encoding of 7 clusters of amino acids based on electrochemical properties.
    - RCNN is a multi-layer network, with one layer consisting of convolutions -> max pooling -> bidirectional GRU -> output, with a residual connection between the convolutional pooling output and the GRU output. The final output is obtained by a convolution layer followed by global average pooling.
  * **Key Strengths**:
    - Sensible Architecture, Robust Comparisons, & Strong results
    - Comparison on 3 versions of task -- binary, multiclass, and K_d prediction.
  * **Key Weaknesses**:
    - No ablation studies on architecture.
  * **Open Questions**:
    - Has this been tried with protein pre-training systems? If not, why not?
  * **Extensions**:
    - Extend to protein interface prediction?
    - Extend to `n`-ary interactions?
  * **How to learn more**:
    - Need to look at more citing papers to see if there is a better benchmarking source.


## [Generative probabilistic biological sequence models that account for mutational variability](https://www.biorxiv.org/content/10.1101/2020.07.31.231381v1.abstract)
  * **Logistics**:
    - Weinstein EN, Marks DS. Generative probabilistic biological sequence models that account for mutational variability. bioRxiv. 2020 Jan 1.
    - 0 as of 9/17/2020
    - Harvard Medical School Departments of Biophysics & Systems Bio
    - Time Range: 09/17/2020 (9:52am - 11:02am)
  * **Summary**:
    - _Single Big Problem/Question_ Large-scale sequencing is really informative, but current methods use multiple sequence alignment, which is unreliable and vulnerable to statistical anomalies which limit their utility.
    - _Solution Proposed/Answer Found_ They introduce the MuE emission distribution which is a generalization of classical models and accounts for possible sequence substitutions, insertions, and deletions, and allows the application of generic models to model sequence data (e.g., neural networks).
    - _Why hasn't this been done before?_ The utility of probabilistic sequence modeling is rising rapidly given increase in available computational power and the importance of being able to integrate arbitrary ML models into these pipelines is also more important than ever. 
    - _Experiments used to justify?_ They show theoretically that MuE generalizes classical models, and empirically that H-MuE models can infer representations indicative of immune repertoires, functional proteins, and forecast evolution of pathogens. Specifically: 
    - _Secret Terrible Thing_ It's 59 pages.
    - 3 most relevant other papers:
      1)
      2)
      3)
    - Warrants deeper dive in main doc? Yes -- I'm bleeding some of that through to here.
  * **Detailed Methodology**:
    Generative models readily exist for continuous state vectors in the sciences, but biological sequences are tricky. In order to generalize continuous models to biological sequences, people often rely on an emission distribution, which takes continuous vectors and generates discrete outputs. Good emission distributions should (1) generate the right kind of data, (2) capture common variability in the data, and (3) be convenient for modelling.
    
    Mutational Emission (MuE) distribution is a generative model for biological sequences that accounts for biological variability. Hierarchical MuE (H-MuE) models use continuous state-space models + the MuE distribution. MuE satisfies all 3 goals -- it is designed specifically w/ biological sequences in mind, and also has an analytically tractable and differentiable likelihood function, enabling inference of H-MuE parameters from data via SGD & variational inference.
    
    What is the MuE distribution? Two step generative process: First, a "latent" sequence is drawn `v_i \sim p(v | \theta)`, where `i` is not the index sequentially but instead the index into the dataset of sequences to model (e.g. `y_i \in \{y_1, \ldots, y_N\}`). This `v_i` is a probability distribution over an "ancestral sequence" logo, and is a matrix of shape `M \times D`. -- what is `M`? Ancestral sequence length. What is `D`? Sequence element vocabulary. `y_i` is then generated from `MuE(x_i = \softmax(v_i), c, a, \ell)`, where `c, a, \ell` describe, respectively, insertion sequence probabilities, indel probabilities, and substitution probabilities. Softmax is over the `D` dimension, so `D` must be the sequence vocabulary size. The goal, then, when given a set of sequences `y_i`, is to learn the parameters of the `H-MuE` model via variational inference to maximize the likelihood of the data given the parameters (or the posterior likelihood, depending on framing, presumably).
    
    But this still hasn't answered our prompting question. What is the MuE distribution? It's a structured HMM, with initial transition vector `a^{(i)}`, transition matrix `a^{(t)}` (I think this superscript notation is not meant to be index, but instead different variables all lumped together under `a` -- e.g., I don't think `a^{(i+1)}` is defined), and discrete emission matrix `e = (\xi \cdot x + \zeta \cdot c) \cdot \ell`, where `\xi` and `\zeta` are fixed constant matricies of shape `K \times M` and `K \times (M + 1)`. `c`, the previously described insertion sequence probability parameter, is of shape `(M+1) \times D`, and `\ell`, the previously defined substitution parameter, has shape `D \times B`. `c` and `\ell` are also defined to be row normalized (rows sum to 1) like `x`. Here, we've introduced two new parameters, `K` and `B` -- `K = 2(M+1)` is the size of the Markov chain state space, and `B` is the alphabet size of the observed sequence. Recall that `x`, being the ancestral sequence, has shape `M \times D`, so `\xi \cdot x` has shape `K \times D`, and `\zeta \cdot c` also has shape `K \times D`, so `e = (\xi \cdot x + \zeta \cdot c) \cdot \ell` has shape `K \times B`. This makes sense, as it is tracking the emission probability for each state in the markov model.
    Some additional restrictions: `\xi` and `\zeta` are both indicator matrices, where `\xi_{k, m} = \delta_{k, 2m}` and `\zeta_{k, m} = \delta_{k, 2m-1}`. What does this imply? This means that
    
    `(\xi \cdot x)_{i, j} = \sum_{r = 1}^{M} \xi_{i, r} x_{r, j} = \sum_{r = 1}^{M} \delta_{i, 2r} x_{r, j} = \begin{cases} 0 & \text{if } i \bmod 2 = 1 \\ x_{i / 2, j} & \text{otherwise.} \end{cases}`
    
    and 
    
    `(\zeta \cdot c)_{i, j} = \sum_{r=1}^{M+1} \zeta_{i, r} c_{r, j} = \sum_{r=1}^{M+1} \delta_{i, 2r-1} c_{r, j} = \begin{cases} 0 & \text{if i \bmod 2 = 0 \\ c_{\frac{i+1}{2}, j} & \text{otherwise.}`
    
    Thus, their sum is going to alternate along dim-0, first capturing the slice `x_{i, :}`, then capturing the slice `c_{i+1, :}`, and so on. As dim-0 is the state-space dimension, this means that there will be a state in this HMM for (1) each position in the ancestor sequence, and (2) a possible insertion in each gap in the ancestor sequence. Hence, the full state space is exactly `M` (positions) + `M-1` internal gaps, plus `2` boundary gaps, for a total of `2M + 1`. This is... almost right. Where's the extra `1` coming from? Current hypothesis, this is a typo -- in reality, `K = 2M + 1`. Or, perhaps the extra state is the stop state? But given by current construction that last row would be all zeros, I'm not sure I understand how that would be captured.
    
    Last restriction -- how do transitions work? Well, `a^{(t)}` must satisfy the restriction that `a^{(t)}_{k, k'} = 0` for all `k, k'` such that (1) state `k` is accessible from the initial state, and (2) `k' + k'\bmod 2 - k + k\bmod 2 \le 0`. What does this mean? We will find out later.
    
    A few questions:
      0) Why the separation between "insertion sequence probabiliteis" and "indel sequence probabilities"? Is the former _what_ is inserted, and the latter _whether something is inserted/deleted_?
      1) What does "size of the Markov chain state space" mean here? Traditional definition implies this is the size of the emission vocabulary -- i.e., the number of states. That doesn't feel right here -- `K = 2(M+1)` is more states than the output sequence is likely to have. Is this simply larger to account for insertions?
  * **Pros**:
    - List of big pros of the paper
  * **Cons**:
    - Not the clearest.
  * **Open Questions**:
    - List of open questions inspired by this paper
  * **Extensions**:
    - List of possible extensions to this paper, at any level of thought-out.
  * **How to learn more**:
    - List of terms/concepts/questions to investigate to learn more about this paper.

## [Graph Meta Learning via Local Subgraphs](https://arxiv.org/pdf/2006.07889.pdf)
  * **Logistics**:
    - Huang, Kexin, and Marinka Zitnik. "Graph Meta Learning via Local Subgraphs." arXiv preprint arXiv:2006.07889 (2020).
    - 0 (as of 6/2020)
    - Harvard
    - Time Range: 16:49 - 17:31 (day 1); START - END (day 2)
  * **Summary**:
    - Rapid adaption of a trained model for graph-structured data is difficult, and meta-learning on graph structured data is under-explored. In addition, current methods for this problem are limited in scope and do not scale well to large graphs.
    - They propose "G-META", a novel meta-learning approach for graphs that uses "local subgraphs" to transfer subgraph-specific information and make the model learn the essential knowledge faster via meta-gradients (WDTMT?).
    - Graph neural networks are relatively new.
    - _Experiments used to justify?_
      1) Experiments spanning 7 datasets and 9 baselines. 
    - _Secret Terrible Thing_ What is the "secret terrible thing" of this paper?
    - 3 most relevant other papers:
      1) Maybe one of these? https://arxiv.org/abs/1905.09718, https://arxiv.org/abs/1912.09867, https://arxiv.org/abs/1909.01515, https://papers.nips.cc/paper/8389-learning-to-propagate-for-graph-meta-learning.pdf
      2) Maybe one of these? https://www.cse.wustl.edu/~muhan/papers/KDD_2017.pdf, https://arxiv.org/abs/1810.00826, https://papers.nips.cc/paper/7763-link-prediction-based-on-graph-neural-networks.pdf
      3) Maybe https://arxiv.org/abs/1905.07953, https://iclr.cc/virtual_2020/poster_BJe8pkHFwS.html#:~:text=We%20propose%20GraphSAINT%2C%20a%20graph,or%20edges%20across%20GCN%20layers.
    - Warrants deeper dive in main doc? (options: No, Not at present, Maybe, At least partially, Yes)
  * **Detailed Methodology**:
    Firstly, note that they restict their notion of GNN to specifically message-passing NN.
    
    G-META can tackle 3 distinct meta learning problems. 
      1) Single graph, disjoint labels (e.g., given a graph, learn how to predict labels `l_1^a, \ldots, l_n^a` at train time, then adapt this into learning how to predict labels `l_1^b, \ldots, l_m^b` at test time. Same graph means same node features, but new labels.
      2) Multiple graphs, shared labels (e.g., given graph 1, learn how to classify nodes in a certain way, then in graph 2 perform the same task). My Question: Why is this meta-learning? If at train time you learn over a bunch of graphs, which are drawn from the same distribution as the test graph in an iid manner, this just seems like learning.
      3) Multiple graphs & disjoint labels. Fig suggests that rather than this being just 1 & 2 in concert, this is more like I have lots of graphs on which I predict labels `l_j^a` but at test time I want to adapt to predicting `l_j^b` on the same collection of graphs.
    G-META formulates this learning task as a subgraph prediction problem. Rather than classifying nodes, nodes are realized as embedded subgraphs, and these are used in the learning formulation. The authors show theoretical properties regarding how the use of local subgraphs can preserve certain inequality relationships about graphs ensuring that minimal informaiton is lost in this approach. This justifies its use over the entire graph, and enables this as a viable way to increase scalability.
  * **Pros**:
    - List of big pros of the paper
  * **Cons**:
    - List of big cons of this paper
  * **Open Questions**:
    - List of open questions inspired by this paper
  * **Extensions**:
    1) Why use single subgraphs? Why not a multi-scale resolution that captures both the local context and the global context of a node in its representation as a classification task and in meta-learning transfer?
  * **How to learn more**:
    - List of terms/concepts/questions to investigate to learn more about this paper.

# Clinical Representation Learning / Pre-training
## [Hierarchical Attention Propagation for Healthcare Representation Learning](https://dl.acm.org/doi/abs/10.1145/3394486.3403067)
  * **Logistics**:
    - Muhan Zhang, Christopher R. King, Michael Avidan, and Yixin Chen. 2020. Hierarchical Attention Propagation for Healthcare Representation Learning. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 249–256. DOI:https://doi.org/10.1145/3394486.3403067
    - 0 (9/1/2020)
    - Washington University in St. Louis
    - Time Range: 11:02am - 11:30 am (28 min; some time spent refining scaffold).
  * **Summary**:
    - _Single Big Problem/Question_ Medical ontologies are useful in many contexts, but require embeddings to use effectively, and current models to produce such embeddings can be improved.
    - _Solution Proposed_ This model leverages a hierarchical attention model to leverage the hierarchical structure of medical ontologies to learn embeddings for nodes that are not only dependent on their immediate node identity but also on the full hierarchy of ancestor nodes.
    - _Why hasn't this been done before?_ Prior work has explored attention over ancestors (in particular, GRAM does this), but these model have used ancestors as an _unordered set_ of ancestors, rather than a hierarchical ontology of ancestors  descendents (which GRAM ignores) in its own right, which this paper explores. In addition, graph neural networks and attention architectures are relatively new and still under active development.
    - _Experiments used to justify?_
      1) 2 sequential procedure/diagnosis prediction tasks (over the ACTFAST and MIMIC-III datasets as sequences of codes)
    - _Secret Terrible Thing(s)_
      1) Comparisons are a not on GRAM's home turf, and they need to greatly restrict the number of layers of the hierarchy they use to acheive great results. Restricting the # of ontological layers _should_ cut the benefits of their approach, as the ontological complexity of the ancestor/descendent ontology is much reduced. But, this also points to key con #3 -- their model can't actually effectively leverage that information, I expect.
    - 3 most relevant other papers:
      1) [GRAM](https://dl.acm.org/doi/abs/10.1145/3097983.3098126?casa_token=vozPkFsiuhYAAAAA:IFZJaw00nViPfNhYTp98XPnVa-DkeD_SCybb-FSxwD4UCvQ_comFkuz4UkoK-zuJvRAL-PW8mKZ-TA), Edward Choi et al, KDD '17, 81 citations. 
        - They improve on this model
      2) [Belief Propagation Algorithm](https://www.aaai.org/Papers/AAAI/1982/AAAI82-032.pdf), Judea Pearl, AAAI '82, 1004 citations.
        - They build on this algorithm to operate over the tree of ancestors.
      3) None
    - Warrants deeper dive in main doc? Not at present.
  * **Detailed Methodology**:
    HAP breaks down attention propagation into two message-passing phases:
      1) Bottom-up propagation: the embedding for node i is updated via attention computation over its immediate _ancestors_ (& itself).
      2) Top-down propagation:  the embedding for node i is updated via attention computation over its immediate _descendents_ (& itself).
    They do this in sequence, not in parallel -- e.g., for round one of propagation, they update the graph downwards, using the updated node values for each subsequent layer--in particular, the random initialization embeddings are used as the raw ancestor inputs only for the lowest layer, and subsequent layers up towards the root (top) node use the embeddings computed over the attention over those nodes' ancestors).
    
    Final modelling is done via an RNN over sequences of medical codes, with each timepoint represented as a nonlinear transform of the sum of the embeddings at a particular visit.
  * **Pros**:
    - They prove that HAP is strictly more expressive than GRAM.
    - They compare across two datasets, and compare to GRAM, a relevant baseline, plus some others.
  * **Cons**:
    - Utility of HAP is strictly tied to utility of GRAM -- HAP is exclusively an improvement to GRAM.
    - Each propagation (down and up) needs to process over the entire graph sequentially. This has a high time complexity. They claim GRAM can have a higher time complexity, but I'm not sure if this is true without reading GRAM -- my instinct is that it is not.
    - Information from lower layers in the hierarchy will decay over multiple attention computations prior to reaching the top of the graph.
    - They don't directly compare to GRAM on GRAM's published tasks -- GRAM reduces comparison space to CCS concepts -- they use full ICD codes. This may bias comparison.
    - Results aren't super compelling.
  * **Open Questions**:
    - Are there really no interleaving other improvements to GRAM? GRAM came out in 2017.
    - What ontology do they use?
  * **Extensions**:
    - Solve con 3: decay of information across layer.
  * **How to learn more**: N/A

## [HiTANet: Hierarchical Time-Aware Attention Networks for Risk Prediction on Electronic Health Records](https://dl.acm.org/doi/abs/10.1145/3394486.3403107)
  * **Logistics**:
    - Junyu Luo, Muchao Ye, Cao Xiao, and Fenglong Ma. 2020. HiTANet: Hierarchical Time-Aware Attention Networks for Risk Prediction on Electronic Health Records. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '20). Association for Computing Machinery, New York, NY, USA, 647–656. DOI:https://doi.org/10.1145/3394486.3403107
    - \# of citations of this work & as-of date
    - Penn State & IQVIA
    - Time Range: 11:47am - 12:12pm (not finished yet)
  * **Summary**:
    - _Single Big Problem/Question_ Disease progression / risk progression is important, but traditionally done in a way that assumes stationarity (WDTMT?) and homogeneity across patients, which are unrealistic assumptions.
    - _Solution Proposed_ The authors use a hierarchical, time-aware attention network (HiTANet) for progression/risk prediction leveraging time information at both local and global stages via a transformer with a time-aware key-query attention mechanism (WDTMT?)
    - _Why hasn't this been done before?_
      1) Attention networks are new?
    - _Experiments used to justify?_
      1) Evaluated on three real-world datasets, showing gains of over 7% in F1 score on all datasets. Task is disease prediction task (COPD, Heart Failure, Kidney disease), against a variety of baselines including attention models, RNNs, baselines, and time-based models.
    - _Secret Terrible Thing_ What is the "secret terrible thing" of this paper?
    - 3 most relevant other papers:
        1) [Dipole](https://dl.acm.org/doi/abs/10.1145/3097983.3098088?casa_token=Cl3YdRF93SwAAAAA:SMVqk6aVi_Z_B9RhhTKdYJRq0l7rbiTMnpyUx2uRJ1MBiAkV5giWPETSu8RyBTyxtsYfNRjUdJF4qQ), Ma et al, KDD '17. This is both a comparison, and a methodological motivator
      There are also 2 clear topic areas of relevance: (todo: add links to these papers)
        1) Attention mechanisms for risk prediction (e.g., Retain, Dipole)
        2) Time-aware models for disease prediction (e.g., T-LSTM, Retain EX, TimeLine, and ConCare)
    - Warrants deeper dive in main doc? At least partially.
  * **Detailed Methodology**:
    HiTANet consists of two stages:
      1) Local evaluation stage:
         This stage deals with C1 (that historical information does not decay monotonically) via a Transformer, which learns time-aware attention weights for individual visits. It learns local attention weights for visits and an overall representation for each patient.
      2) Global synthesis tage
         This stage deals with C2 (that the importance of historical information varies across patients), by utilizing the per-patient representation produced in stage #1 via a transformer that generates a global attention weight for each visit. WDTMT?
         
    This paper still works with EHR data as sequences of codes -- no numerical data leveraged in this work.
      
    Technical details on stages:
      1) Local:
        1) Embed visit codes into embedding space
        2) Embed time delta between visit and prior visit into latent space via 1-hidden layer NN. The time embedding layer has a square operation to ensure that the system moves rapidly away from zero as time-points grow larger, followed by a tanh interaction. This doesn't seem well justified to me, and is not ablated.
        3) combine visit vector & time vector (vector containing time delta between visits, in days), 
  * **Pros**:
    - Lots of baselines / metrics
  * **Cons**:
    - No GRU-D, which seems a natural model given their framing, possibly? This is likely not included as their focus is strongly at the "visit sequence" level, not the "measurement sequence" level. Nonetheless, their method may be valuable in that context as well!
    - No variance reported.
    - Their critical claim that "existing models assume stationary information decay" isn't terribly well justified/explained. Maybe this is clear for those who have read the relevant related lit, but not for me.
    - No validation of use of square operation.
    - Relative position representations seems more justified than this use of continuoust timepoint embeddings. To further ensure locality is respected, though, they do use "local-biased attention", via dipole (WDTMT?).
  * **Open Questions**:
    - Authors state: "However, existing studies implicitly assume the stationary progression during each time period, and thus take a homogeneous way to decay the information from previous time steps for all patients in their models." WDTMT? ITT?
  * **Extensions**:
    - Can these ideas be relevant to the appropriate use of sporadically measured labs in an inpatient context (within a single visit)?
    - Would relative position representations (embed time delta directly) be a better vehicle here? Possibly in concert with changepoint detection?
  * **How to learn more**:
    - Read Dipole 
    - Read papers in category 2, above


# Unstructured
## [Predicting What You Already Know Helps: Provable Self-Supervised Learning](https://arxiv.org/abs/2008.01064)
  * **Logistics**:
    - Lee JD, Lei Q, Saunshi N, Zhuo J. Predicting What You Already Know Helps: Provable Self-Supervised Learning. arXiv preprint arXiv:2008.01064. 2020 Aug 3.
    - 1 09/16/2020
    - Princeton, UT Austin.
    - Time Range: 09/16/20 (10:00am - 10:27am), 09/17/20 (9:24am - 9:44am)
  * **Summary**:
    - _Single Big Problem/Question_ Why does self-supervised pre-training help?
    - _Solution Proposed/Answer Found_ The authors quantify how approximate independence between pre-training task allows one to learn representations that can solve downstream tasks with significantly reduced sample complexity
    - _Why hasn't this been done before?_ Pre-training is exploding over the last 2 years, leading to greater demand for theoretical models explaining this phenomena
    - _Experiments/theorey used to justify?_ They present theory based on relating conditional independence of input, pre-training task, and downstream task establishing their claims, and validate this theory with simulation experiments and experiments on the SST dataset and the Yearbook dataset.
      1) List of experiments used to justify (tasks, data, etc.) -- full context.
    - _Secret Terrible Thing_ Their theory only covers the case of (1) a single `Y` and (2) a fixed `\psi` representation of `X_1`, when in reality we'd actually work with a (potentially variable) representation of `X`. This difference is, in many cases, critical, I think.
    - 3 most relevant other papers: Unknown
    - Warrants deeper dive in main doc? Not at present.
  * **Detailed Methodology**:
    Authors begin with the following remark: In certain cases, given `X_1` being the underlying object (e.g., an image of a person with the face occluded), `X_2` being the self-supervised target (e.g., the true face of an image), and `Y` being the target of interest in a downstream task (e.g., the person's identity). In this case, provided you know `Y`, you can infer `X_2` reasonably well independent of `X_1`. This therefore implies that `X_1 \propto X_2 | Y`, and further implies that to predict `X_1 \to X_2`, we internally must be travelling through `Y`. That last leap, I think is a little wrong. For example, suppose in my training set there are only 2 people, despite the fact that `Y`s label space is actuall 1000 people. In that setting, it is true that `X_1 \propto X_2 | Y`, but it is also true that there is a "simpler" variable `B` flagging which of the two people the person is that also satisfies that description, and `B` does not fully inform `Y` (they account for this in their theory by commenting on the necessary rank of the covariance matrix between `X_2` and `Y`). Another flaw with this toy example -- in my final task, I'm not actually trying to go `X_1 \to Y`; I'm really trying to go `X_2 \to Y`, b/c that's the most informative section of the image. So, the fact that `X_1 \to X_2` (maybe) has to go through `Y` doesn't seem relevant. More relevant than the `X_1 \to Y` part of this proposed flow is the `Y \to X_2` part. This maybe implies using invertible NNs for the pre-training task, then reversing them in fine-tuning would be beneficial?
    
    They go through a number of proofs, quantifying how much sample complexity they need to learn a linear downstream task model atop a fixed representation `\psi` learned from `X_1 \to X_2` in various cases. They cover:
      * Gaussian Random Variables (Exact solution, Exact assumptions)
      * General Random Variables (Exact assumptions & Approximate assumptions, sufficiently quantified -- one particular point they cover is when you need to introduce another latent variable to "fill out" the covariance relationship between X_2 and the target)
      * General Function Classes
      
    These analyses are always capped by possible performance of `X_1 \to Y` and `X_1 \to X_2`; however, under assumption of CI, these two quantities are related through their analyses, I believe. E.g., if `X_1` is not at all predictive of `Y` (if `X_1 \perp Y`), and CI holds, then `X_1 \perp X_2`? That's not quite true -- but suppose that `\Cov(X_2, Y)` is full rank -- then it may hold? B/c otherwise any inference about `X_2` from `X_1` would also be informative about `Y`.
  * **Pros**:
    - Nice theoretical framework, subject to concern that in reality, more common situation is `X_1 \to X_2` as pre-training/pretext, and `(X_1, X_2) \to Y` as fine-tune, or `X_1 \to X_2` on domain `D_PT`, then `X_1 \to Y` on domain `D_FT`, both of which are different.
    - Extensive theoretical analyses, across different levels of assumption matching. Quantification of how much assumption mismatch hinders result discovery.
    - Multi-modal real-world experiments.
  * **Cons**:
    - Theoretical framework is adjacent to real-world setting. They don't examine this discrepancy or comment on it, as far as I can tell.
  * **Open Questions**:
    - How does this theory extended to cases where there are many `Y`s of interest, as in the case of BERT? This can be partially handled 
  * **Extensions**:
    - List of possible extensions to this paper, at any level of thought-out.
  * **How to learn more**:
    - List of terms/concepts/questions to investigate to learn more about this paper.
  * **Other Notes**:
    - pre-text task = pre-training task
