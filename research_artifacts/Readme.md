# Research Artifacts Notes

In this document I'll summarize discrete research artifacts I've read, leaving notes.

## Acronyms
  1. WDTMT?: "What does this mean, technically?"
  2. ITT?: "Is this true?"

## Expected Format
### Summary v1 (~ 15 min)
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

### Summary v2 (~ 30 min)
Goal is to give a full, complete summary of the paper.
```
## [Paper_Title](paper_link)
### Logistics:
  - Citation (key points: publication venue, date, authors)
  - \# of citations of this work & as-of date
  - Author institutions
  - Time Range: DATE (START - END)
### Summary:
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
### Detailed Methodology:
  Detailed description of underlying methodology
### Key Strengths:
  - List of big pros of the paper
### Key Weaknesses:
  - List of big cons of this paper
### Open Questions:
  - List of open questions inspired by this paper
### Extensions:
  - List of possible extensions to this paper, at any level of thought-out.
### How to learn more:
  - List of terms/concepts/questions to investigate to learn more about this paper.
```

# Uncategorized
##[Topological Autoencoders](https://arxiv.org/pdf/1906.00722.pdf)
### Logistics:
  - Moor M, Horn M, Rieck B, Borgwardt K. Topological autoencoders. InInternational Conference on Machine Learning 2020 Nov 21 (pp. 7045-7054). PMLR.
  - 11 (12/2020)
  - ETH Zurich
  - Time Range: 12/22/20 (16:23 - END)
### Summary:
  - _Single Big Problem/Question_ It would be valuable if we could learn autoencoders that perserved topological features of point clouds, but preserving topology directly is challenging as topological criteria are not differentiable natively.
  - _Solution Proposed/Answer Found_ The authors propose a vehicle to obtain gradients of topological signatures, making it possible to employ topological constraints while training deep neural networks and building topology-preserving autoencoders.
  - _Why hasn't this been done before?_ Why has nobody solved this problem in this way before? What hole in the literature does this paper fill? (1 sent).
  - _Experiments used to justify?_
    1) List of experiments used to justify (tasks, data, etc.) -- full context.
  - _Secret Terrible Thing_ Why do I want to learn an autoencoder that perserves topology? Is this for compression? What if I think my input metric is really stupid, so my topology is bad, and I want to learn a \emph{better} topology? Also, they only suppor the simplest of topological features?
  - 3 most relevant other papers:
    1)
    2)
    3)
### Detailed Methodology:
  Persistent homology extends simplicial homology from underlying manifolds/simplices to point clouds (e.g. `X = \{x_1, \ldots, x_n\} \in \R^d`, and a metric `\dist: X \times X \to \R`). How? By defining simplices across multiple scales of the metric `\dist` and examining their homological properties. For small (sufficiently) changes of the metric, the simplices don't change, so the homology signatures are continuous w.r.t. the metric. The actual persistent homology signature itself is a collection of points defining regions of the metric threshold where a particular topological feature exists.
  
  For a topology-preserving autoencoder, the authors compute the persistent homology signature of the latent code Z induced by an autoencoder `X \to Z \to \tilde{X}`, and assert that this should be identical to that of `X`. They do this by realizing the persistence diagram as really a selection operation within the all-pairs distance matrix of the dataset, and, given continuity, the selection property will have no gradient so they can just compare distances on all selected edges, which gives clean losses. This is, in essence, a way of training your autoencoder such that it perserves the relative distance relationships among the critical nodes in the dataset (and, by unioning the edges between those critical for X and Z, it also will inherently push the edge-sets to be the same, by making the edges of the unimportant Z selections change to mirror their unimportant state in X).
  
  The authors also show some properties re stability of this approach.
### Key Strengths:
  - List of big pros of the paper
### Key Weaknesses:
  - List of big cons of this paper
### Open Questions:
  - How does this work given the persistence diagram won't change much given small changes in metric or representation? Ahh, that's actually a misunderstanding. As the persistence diagram captures bounds on the threshold directly, small changes in the metric will change that threshold exactly, so it does have a continuous gradient. In fact, instead of being nearly zero everywhere, it will be identity nearly everywhere, with more abrupt changes at various points.
### Extensions:
  - List of possible extensions to this paper, at any level of thought-out.
### How to learn more:
  - List of terms/concepts/questions to investigate to learn more about this paper.

## [WILDS: A Benchmark of in-the-Wild Distribution Shifts](https://arxiv.org/abs/2012.07421)
### Logistics:
  - Koh PW, Sagawa S, Marklund H, Xie SM, Zhang M, Balsubramani A, Hu W, Yasunaga M, Phillips RL, Beery S, Leskovec J. WILDS: A Benchmark of in-the-Wild Distribution Shifts. arXiv preprint arXiv:2012.07421. 2020 Dec 14.
  - Stanford, Berkeley, Microsoft, Cornell, CalTech
  - Time Range: 12/22 (15:57 - 16:21) -- stopping early as I think I have the feel for the paper and it isn't immediately relevant to me beyond that.
### Summary:
  - _Single Big Problem/Question_ Distribution shifts cause major problems for ML in the real world, but lack organized, realistic benchmarks on which researchers can evaluate novel paradigms for learning under shifts.
  - _Solution Proposed/Answer Found_ The authors propose WILDS, a benchmark of in-the-wild distribution shifts spanning diverse modalities and applications. Provides real-world, domain-oriented train/test splits and evaluation metrics.
  - _Why hasn't this been done before?_ This is a big project that takes a lot of organizational effort, and is only relevant at the moment when ML is beginning to be used sufficiently in the real-world that issues w/  dataset shift are causing real problems. To surmount the logistical work and domain-expertise required to create something like WILDS, prior efforts also have generally introduced synthetic examples of dataset shift, which are not necessarily representative of true domain shift.
  - _Experiments used to justify?_ Not super applicable, but the authros do profile ERM, DeepCoral, IRM, and GroupDRO on their benchmark.
  - _Secret Terrible Thing_
    Two things: First, I don't like the static train/test split. _Especially_ in the context of domain shift, where train is not iid of test, repeated train/test sampling is essential to get a strong estimate of mean, I think. Depending on the # of domains, this may or may not be a big problem. Second, I think we _must_ assume that D_train and D_test are either iid samples from some domain generative process, or that the pair (D_train, D_test) is a sample from some domain adaption generative process D_gen (in either case), and that we have some way of further _simulating_ sampling from D_gen (e.g., subsampling the train set). In this way, learning how to solve domain generalization on D_train helps us on D_test. This can only work, however, if their datasets are truly constructed in this way, which does not appear to be the case.
  - 3 most relevant other papers:
    1)
    2)
    3)
### Detailed Methodology:
  They define "Domain generalization" to be when the domains at train time are non-overlapping with those at test time (e.g., train on MIMIC, test on eICU), and "subpopulation shift" when the domains are the same but the relative makeup may be different (e.g., we train on mostly white people but want to do well on all people). Note these are both different from "Domain Adaption", where we know D_train and D_test and want to adapt from D_train to D_test explicitly. They simulate Domain generalization as domain adaption, though, by adding OOD validation sets for tuning.
### Key Strengths:
  - List of big pros of the paper
### Key Weaknesses:
  - List of big cons of this paper
### Open Questions:
  - List of open questions inspired by this paper
### Extensions:
  - List of possible extensions to this paper, at any level of thought-out.
### How to learn more:
  - List of terms/concepts/questions to investigate to learn more about this paper.


## [Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Multi-Similarity_Loss_With_General_Pair_Weighting_for_Deep_Metric_Learning_CVPR_2019_paper.pdf)
### Logistics:
  - CWang X, Han X, Huang W, Dong D, Scott MR. Multi-similarity loss with general pair weighting for deep metric learning. InProceedings of the IEEE Conference on Computer Vision and Pattern Recognition 2019 (pp. 5022-5030).
  - 110 (as of 12/2020)
  - Malong Technologies, China
  - Time Range: 12/17/20 (09:41 - 10:42)
### Summary:
  - _Single Big Problem/Question_ A nubmer of pair- or tuple- based Deep Metric Learning (DML) loss functions have been proposed, but no general unfiying framework exists across these losses. 
  - _Solution Proposed/Answer Found_ The authors establish the General Pair Weighting (GPW) framework, which recasts the sampling problem in DML into  a unified view of pair weighting through gradient analysis. This allows various pair- or tuple-based loss functions to be discussed comprehensively, and allows one to propose the new _multi-similarity loss_ (MS loss), which is a more principled loss than the existing solutions.
  - _Why hasn't this been done before?_ DML is an active area of research, but is still driven by empirical findings, rather than new theory that merges older metric-learning theory with new NN based ideas. This paper helps merge a variety of recent empirical approaches under a novel theoretical lens.
  - _Experiments used to justify?_  The MS loss also obtains new state-of-the-art performance on four image retreival benchmarks, compared to recent approaches such as ABE and HTL by large margins.
    Benchmarks:
        1) CUB200
        2) In-shop Clothes Retrieval dataset
        3) ?
        4) ?
     Models:
        1) ABE
        2) HTL
  - _Secret Terrible Thing_ What is the "secret terrible thing" of this paper?
  - 3 most relevant other papers:
    1)
    2)
    3)
### Detailed Methodology:
  #### General Pair Weighting
  Given `\vec x_i \in \R^d` an instance vector, `\mat X \in \R^{m \times d}` an instance matrix, and `\vec y \in \{1, 2, \ldots, C}^m` a label vector. `\vec x_i` is projected ono an `\ell`-dimensional unit sphere by a neural network `f(\cdot ; \vec \theta): \R^d \to S^\ell` parametrized by `\vec \theta`. Formally, the similarity of two samples is defined as `S_{i,j} := \ip{f(\vec x_i ; \vec \theta), f(\vec x_j ; \vec \theta)}` (cosine similarity given unit sphere). Let `\mat S` be the `m \times m` matrix with `S_{i, j}` as its elements.
  
  Any pair-based loss `\mathcal L` can be formulated as a function of `\mat S, \vec y`. How? Consider the gradients that would be used to optimize `\mathcal L`: `\left. \pd{\mathcal L(\mat S, \vec y)}{\vec theta} \right|_t = \left.\pd{\mathcal L(\mat S, \vec y)}{\mat S} \right |_t \left \pd{\mat S}{\vec \theta} \right|_t` which in turn is equal to `\sum_{i=1}^m \sum_{j=1}^m \left.\pd{\mathcal L(\mat S, \vec y)}{S_{i,j}}\right|_t \left \pd{S_{i, j}}{\vec \theta} \right|_t`. 
  
  This can be re-formulated into a function `\mathcal F` whose gradient w.r.t. `\vec \theta` at the `t`-th iteration is computed exactly the same as Eq. 1: `\mathcal F(\mat S, \vec y} = \sum_{i=1}^m \sum_{j=1}^m \left.\pd{\mathcal L(\mat S, \vec y)}{S_{i,j}}\right|_t S_{i, j}`. Note that for this to be true, we must regard `\left.\pd{\mathcal L(\mat S, \vec y)}{S_{i,j}}\right|_t` as a _constant scalar_ independent of `\vec \theta` -- e.g., `\mathcal F` is really `\mathcal F_t`, which is a weighted sum whose (constant) weights happen to be equal precisely to `\left.\pd{\mathcal L(\mat S, \vec y)}{S_{i,j}}\right|_t`.
  
  For a pair-based loss (as we want to push positive pairs closer and negatives apart) we can assume `\left.\pd{\mathcal L(\mat S, \vec y)}{S_{i,j}}\right|_t \ge 0` for a negative pair and `\left.\pd{\mathcal L(\mat S, \vec y)}{S_{i,j}}\right|_t \le 0` for a positive pair. Thus, we can reformulat `\mathcal F` in the form of pair weighting as follows:
  
  `\mathcal F = \sum_{i=1}^m \left(\sum_{j | \vec y_j \neq \vec y_i}w_{ij} S_{ij} - \sum_{j | \vec y_j = \vec y_i}w_{ij} S_{ij}\right)`
  where `w_{ij} = \abs{\left.\pd{\mathcal L(\mat S, \vec y)}{S_{i,j}}\right|_t}` (really, this should be `w_{ij}^{(t)}` as it does depend on iteration `t`). Thus, we can formulate any pair-based loss as a general pair re-weighting scheme. 
  
  ##### Re-interpretation under a graph lens
  Under a graph lens, we lack `\vec y` and instead have adjacency matrix `\mat A \in \{0, 1\}^{m \times m}` corresponding to some graph `G`. With this mentality, we can re-interpret `\mathcal F` above as:
  `\mathcal F = \sum_{i=1}^m \left(\sum_{j | \mat A_{i,j} = 0}w_{ij} S_{ij} - \sum_{j | \mat A_{i, j} = 1}w_{ij} S_{ij}\right)`. But this now looks suspiciously like a dot product. With a little re-arranging:
  
  `\mathcal F = \sum_{i=1}^m \ip{\vec w_i \vec S_{i, :}} - 2 * \ip{\mat_A{i, :}, \vec w \odot \vec S_{i, :}}`
  `\mathcal F = \sum_{i=1}^m \ip{\vec 1, \vec w_i \odot \vec S_{i, :}} - 2 * \ip{\mat_A{i, :}, \vec w \odot \vec S_{i, :}}`
  `\mathcal F = \sum_{i=1}^m \ip{\vec 1 - 2\mat_A{i, :}, \vec w_i \odot \vec S_{i, :}}`
  `\mathcal F = (\vec 1_m^T) \cdot \left( [\mat 1_{m \times m} - 2 \mat A] \odot \mat w \odot \mat S\right) \cdot \vec 1_m`
  `\mathcal F = \vec 1_m^T \cdot \left( \mat w \odot \mat S \right) \vec 1_m - 2 \left( \vec 1_m^T \left( \mat A \odot \mat w \odot \mat S \right) \vec 1_m`
  `\mathcal F = \vec 1_m^T \cdot \left( \tilde{\mat A} \odot \mat w \odot \mat S \right) \vec 1_m - \left( \vec 1_m^T \left( \mat A \odot \mat w \odot \mat S \right) \vec 1_m`
  
  Which we can interpret as a difference between the sum of `\mat w \odot \mat S` re-weighted edge weights in the "inverted" adjacency matrix corresponding to the graph `G'` which has edges precisely where `G` does not and the original graph `G`.
  
  #### Multi-similarity Loss
  ##### Different kinds of similarity
  TODO
  
  ##### Loss Formulation.
  `\mathcal L_{\text{MS}} = \frac{1}{m} \sum_{i=1}^m \left ( \frac{1}{\alpha} \log\left( 1 + \sum_{j | y_i = y_j} e^{-\alpha(S_{i,j} - \lambda)}\right) + \frac{1}{\beta} \log \left(1 + \sum_{j | y_i \neq y_j e^{\beta (S_{ij} - \lambda)}\right)\right)`
  ###### Under graph lens
  `\mathcal L_{\text{MS}} = \frac{1}{m} \sum_{i=1}^m \left ( \frac{1}{\alpha} \log\left( 1 + \sum_{j | y_i = y_j} e^{-\alpha(S_{i,j} - \lambda)}\right) + \frac{1}{\beta} \log \left(1 + \sum_{j | y_i \neq y_j} e^{\beta (S_{ij} - \lambda)}\right)\right)`
  `\mathcal L_{\text{MS}} = \frac{1}{m} (\vec 1_m^T) \cdot \left(\frac{1}{\alpha} \log\left(1 + e^{\alpha\lambda}\mat A \odot e^{-\alpha\mat S}\right) + \frac{1}{\beta} \log \left(1 + e^{-\beta\lambda} \tilde{\mat A} \odot e^{\beta \mat S}\right)\right) \cdot \vec 1_m`
  
### Key Strengths:
  - List of big pros of the paper
### Key Weaknesses:
  - List of big cons of this paper
### Open Questions:
  - List of open questions inspired by this paper
### Extensions:
  - List of possible extensions to this paper, at any level of thought-out.
### How to learn more:
  - List of terms/concepts/questions to investigate to learn more about this paper.


## [Learning to Select the Best Forecasting Tasks for Clinical Outcome Prediction](https://proceedings.neurips.cc/paper/2020/file/abc99d6b9938aa86d1f30f8ee0fd169f-Paper.pdf)
  * **Logistics**:
    - Xue Y, Du N, Mottram A, Seneviratne M, Dai AM. Learning to Select Best Forecast Tasks for Clinical Outcome Prediction. Advances in Neural Information Processing Systems. 2020;33.
    - Google
    - Time Range: 12/09/2020 (15:25 - END)
  * **Summary**:
    - _Single Big Problem/Question_ Designing a meaningful set of PT tasks for a given FT context is difficult and can be inefficient to do by hand.
    - _Solution Proposed/Answer Found_ This paper uses meta-learning and bi-level optimization to automatically learn a weighting vector \lambda for a weighted forecasting task used in pre-training.
    - _Why hasn't this been done before?_ Meta-learning to optimize PT tasks combines several new developments, and builds directly on previous literature in a number of ways. Not documenting this fully as I had a project in a similar vein so am confident this timing makes sense.
    - _Experiments used to justify?_ All with MIMIC-III, using the first 48 hours as input, with 48 prediction windows and *no* gap time.
      1) Low Blood-pressure Detection
      2) Imminent Mortality Prediction
      3) Kidney Dysfunction Prediction
      They test on each of these tasks pre-training natively, pre-training with their system, multi-task learning, direct supervised learning, etc.
    - _Secret Terrible Thing_ Only MIMIC-III, only 3 tasks, with no gap times, on first 48 hours. In other words, very limited genearlizability over tasks and data. In addition, very limited learnability of PT task.
    - 3 most relevant other papers:
      1) Our work: https://arxiv.org/pdf/2007.10185.pdf (not cited)
      2) Meta-learning (maybe in particular https://arxiv.org/abs/1803.02999)
      3) https://arxiv.org/abs/1812.00490
    - Warrants deeper dive in main doc? Yes
  * **Detailed Methodology**:
    They adopt a classical meta-learning setup: In the "Inner Loop", they optimize an encoder for their weighted self-supervised forecasting task, and in the "outer loop" this model is transferred and used for supervised prediction, with the weight for the forecasting weighting updated via meta-learning. Similar to many other methods, exactly solving this meta-learning task is challenging, so the authors adopt a first-order approximation, approximating their meta-gradient as the product of the gradient of their validation loss w.r.t. their encoder parameters _at the end of fine-tuning_, times the gradient of their encoder parameters _at the end of pre-training_ w.r.t. the weighting parameter. This effectively ignores the FT learning process in their gradient calculation.
    
    TODO: Probe this methodology in more depth. Seems like they may also be losing the pre-training algorighm dependence as well...
  * **Key Strengths**:
    - Positive Results
    - Nice framing
  * **Key Weaknesses**:
    - Limited generalizability
    - Heavy assumption burden in meta-learning formulation.
  * **Open Questions**:
    - Is learned task weighting super related to measurement frequency?
    - Why does BP prefer Pretrain(down) to Pretrain(All)?
    - Given the extensive overfitting documented in Figure 2, how does their system perform so well in real tests? Early stopping somehow? Seems in direct supervised learning, they ue early stopping, but not for their PT system. Instead they just do 5 epochs.
  * **Extensions**:
    - Whole meta- pre-training project.
  * **How to learn more**:
    - List of terms/concepts/questions to investigate to learn more about this paper.


## [CliniQG4QA: Generating Diverse Questions for Domain Adaptation of Clinical Question Answering](https://arxiv.org/pdf/2010.16021.pdf)
  * **Logistics**:
    - Yue X, Yao Z, Lin S, Sun H. CliniQG4QA: Generating Diverse Questions for Domain Adaptation of Clinical Question Answering. arXiv preprint arXiv:2010.16021. 2020 Oct 30.
    - Ohio State University, Abigail Wexner Research Institute at Nationwide Children's Hospital
    - Time Range: 11/13/2020 (14:58 - )
  * **Summary**:
    - _Single Big Problem/Question_ Clinical QA is important, but current question generation (QG) models are too simplistic and do not generate sufficiently diverse questions, which damages their generalizability to novel contexts.
    - _Solution Proposed/Answer Found_ This paper proposes a framework, CliniQG4GA, which leverages QG to synthesize QA pairs on new clinical contexts, along with a question-phrase-prediction (QPP) seq2seq model that can augment most QG models to increase diversity. Together, these changes better enable domain adaption to new clinical contexts.
    - _Why hasn't this been done before?_ Clinical QA isn't still a huge field, and LMs/effective NLP data augmentation are currently all the rage motivated by improvements like BERT & GPT.
    - _Experiments used to justify?_
      1) They utilize their framework to improve the generalizability of QA models trained on emrQA going towards MIMIC-III as the target domain. Using their framework gives a boost of up to 8\% in exact match stats on the target domain.
    - _Secret Terrible Thing_ 
    - 3 most relevant other papers:
      1) emrQA
      2) [Question Generation](https://www.aclweb.org/anthology/P17-1123/)
      3) Domain adaption works.
    - Warrants deeper dive in main doc? (options: No, Not at present, Maybe, At least partially, Yes)
  * **Detailed Methodology**:
    - They generate a novel set of ~1300 QA pairs on MIMIC notes. Of these, 975 are human verified based on QG models, and 312 are fully new.
    - Their QPP framework takes an answer snippet, and predicts a sequence of question phrase parts (e.g., "What treatment", "How often", "What dosage", etc.). This is still confusing to me, but seems, in essence, to be backing out the logical forms from the questions (or something like that). 
    - TODO: rest of this section.
  * **Key Strengths**:
    - Shows marked improvements over using simple QG methods or directly trying to generalize from emrQA.
  * **Key Weaknesses**:
    - Only 312 of their questions are actually new.
    - No statistical tests compared, and the differences aren't huge between BeamSearch and QPP
    - It isn't clear to me, but they may be generating compound questions???
  * **Other Notes/Questions**:
    - They only use a 5% sample of the emrQA dataset to account for its redundancy... 
    - Interestingly, the model does better on the "Human Generated" samples than it does on the "Human Verified" samples
    - How would their system help on emrQA directly?
    - Why not use QPP + Beam Search?


## [VIME: Extending the Success of Self- and Semi-supervised Learning to Tabular Domain](https://papers.nips.cc/paper/2020/file/7d97667a3e056acab9aaf653807b4a03-Paper.pdf)
  * **Logistics**:
    - Yoon J, Zhang Y, Jordon J, van der Schaar M. VIME: Extending the Success of Self-and Semi-supervised Learning to Tabular Domain. Advances in Neural Information Processing Systems. 2020;33.
    - 0 (11/12/2020)
    - Google, UCLA, Cambridge, Oxford
    - Time Range: 11/12/2020 (10:10 - 10:55)
  * **Summary**:
    - _Single Big Problem/Question_ Self- and Semi-supervised learning frameworks have not been as successful on tabular data as they have on other modalities.
    - _Solution Proposed/Answer Found_ The authors propose VIME, a pre-training method for tabular data revolving around value imputation and mask estimation. In addition, they propose a novel data augmentation method for tabular data specifically designed for sself- and semi-supervised learning frameworks.
    - _Why hasn't this been done before?_ PT is making a big comeback, but limitations of being applicable only to imaging and text data are sharp.
    - _Experiments used to justify?_
      1) Apply their framework to multiple tabular datasets on a genomics dataset (UK Biobank, 400k patients) and a clinical dataset (Prostate Cancer UK + SEER datasets), comparing to a denoising auto-encoder (DAE) model and a [context encoder](https://openaccess.thecvf.com/content_cvpr_2016/html/Pathak_Context_Encoders_Feature_CVPR_2016_paper.html) (2016) model (for their self-supervised framework) and [MixUp](https://arxiv.org/abs/1710.09412) (2017) for their semi-supervised model, as well as supervised baselines.
    - _Secret Terrible Thing_ Their main value-add really seems to be using self-SL and semi-SL together, not their actual framework. I'd be very curious to see how their baselines perform in a joint self- and semi- SL context.
    - 3 most relevant other papers:
      1) Existing tabular pre-training papers ([TabNet](https://arxiv.org/abs/1908.07442) and [TaBERT](https://arxiv.org/abs/2005.08314))
      2) [MixUp](https://arxiv.org/abs/1710.09412)
    - Warrants deeper dive in main doc? (options: No, Not at present, Maybe, At least partially, Yes)
  * **Detailed Methodology**:
    VIME incorporates both a self-supervised and a semi-supervised learning component. This is interesting, and imo (if it were necessary) could weaken the overall pitch for VIME -- if VIME requires you to do an expensive semi-supervised component during fine-tuning, that'd be unfortunate.
      * Self-supervised Component:
        In this component, they introduce two pretext tasks: _feature vector estimation_ and _mask vector estimation_. Their goal is to optimize a pretext model to recover an input sample (a feature vector) from its corrupted variant, at the same time as estimating the mask vector that has been applied to the sample. First, the system generates a binary mask vector according to a uniform distribution. The pretext generator model `g_m` then maps `(\vec x, \vec m) \mapsto \tilde{\vec x} = \vec m \cdot \bar{\vec x} + (1 - \vec m) \cdot \vec x`, where `\bar{\vec x}` is sampled from the iid approximation of the empirical distribution of `\vec x` induced by the empirical marginal distributions of each component separately.
        
        The pre-text task is solved by dividing it into two subtasks. First, the model predicts which features have been masked, and second it predicts the values of the corrupted features. These tasks are solved independently, not jointly. Notably, for the regression recovery task, they solve it with traditional euclidean regression, rather than with a probabilistic regression and joint loss.
      * Semi-supervised Component:
        In this component, the model utilizes the pre-trained encode (`e`) and the mask generator to generate a number of noised versions of the sample, then uses a consistency loss amongst all noised samples in addition to the supervised prediction loss to regularize the model to perturbation. They claim here that their perturbation is "learned" but it really isn't -- it's a static model utilizing uniform masking and empirical marginal sampling. It _could_ be learned, though, which is interesting.
      * Experiments: Note that in all experiments (aside from ablations on the public datasets) they use VIME's self-supervised in _concert_ with the semi-supervised component whereas their baselines don't have that luxury.
        - Genomics: They use a dataset of 400k patients data, consissting of around 700SNPs and 6 blood cell traits from the UK biobank. They treat SNPs as features and predict the 6 blood cell traits while varying the # of labeled points. Their supervised baseline is ElasticNet, rather than an MLP, as this has performed better than neural methods. That aside, they do show performance improvements (reductions in MSE) across all 6 blood types and in comparison to all baselines, fully supervised and self/semi-supervised methods. They skip the DAE here, which is not explained.
        - Clinical: They use 28 clinical features, and predict treatment choice for the UK prostate cancer patients (2 binary prediction tasks). In the UK they have 10k labeled patients, which they augment with 200k unlabeled samples from teh US datasets. They split the UK dataset 50/50 for training + testing. They compare against DAE, Context Encoder, MixUp, and 3 supervised baselines, Logistic Regression, 2-layer Perceptron, and XGBoost, measuring AUROC as their metric. VIME shows improvements over everything. Here, as in before, the general order is self-supervised < SL < semi-supervised < VIME.
        - Three Public Datasets: MNIST (interpreted tabularly), UCI Income, and UCI Blog, split 10/90 labeled, unlabeled. On these datasets, they also perform ablation studies, which help account for some of the limitations in their other comparisons. Namely, they use VIME in 3 manners -- SL only, self-SL only, or Semi-SL only. They find that SL only is comparable to the other SL models, self-SL only is similar to the other self-SL only, and semi-SL only is similar (albeit a bit better) than the semi-SL baseline. But, everything in concert shows synergistic gains. This, to me, suggests that their framework isn't really as important as the gains found in doing semi-SL and self-SL simultaneously.
  * **Key Strengths**:
    - Nice results. Best-in-class on all comparisons.
    - Model is relatively clear and transparent.
  * **Key Weaknesses**:
    - Their main value-add really seems to be using self-SL and semi-SL together, not their actual framework. I'd be very curious to see how their baselines perform in a joint self- and semi- SL context.
    - In the main paper, they don't say what the architecture of their encoder is and how it compares to their baselines? This might be a big factor!
    - Their baselines are pretty simple, and I doubt they are SOTA. Their self-SL baselines are a DAE and a 2016 paper, and their semi-SL baseline is a 2017 paper. Surely there's been more done since then?
  * **Open Questions**:
    - Can the sampling process for the masked values be made more complicated? E.g., why not train a mask recover auto-encoder on the unlabeled data and use that? Maybe it is because we don't want it to be too good.
    - Does it make sense to fully divide the two subtasks? Maybe so, as their encoder is shared, so they're really putting the meat of the solution for both tasks in the encoder.
  * **Extensions**:
    - Learning mask and obfuscation distributions.

## [Deep Contextual Clinical Prediction with Reverse Distillation](https://arxiv.org/abs/2007.05611)
  * **Logistics**:
    - Kodialam RS, Boiarsky R, Sontag D. Deep Contextual Clinical Prediction with Reverse Distillation. arXiv preprint arXiv:2007.05611. 2020 Jul 10.
    - MIT
    - Time Range: 11/5/2020 (10:55 - 11:35)
  * **Summary**:
    - _Single Big Problem/Question_ Make deep learning algorithms match or exceed the performance of simpler baseline models (e.g., linear model).
    - _Solution Proposed/Answer Found_ Authors propose "Reverse Distillation," which pre-trains deep models by using high-performing linear models for initialization. In particular, the authors propose "Self Attention with Reverse Distillation" (SARD), an architecture that combines contextual embeddings, temporal embeddings, self-attention mechanisms, and reverse distillation to offer significant improvements for processing longitudinal insurance claims data.
    - _Why hasn't this been done before?_ It probably has -- unclear yet what is new about this as compared to prior approaches.
    - _Experiments used to justify?_
      Prediction on a) End-of-Life (EoL), b) Surgical Procedure (Surgery), and c) Likelihood of hospitalization (LoH) task on a dataset of 121.6k Medicare patients in OMOP format, with a single train/val/test split. They compare against four baselines: 2 LR models, a self-attention architecture without reverse distillation (this is an ablation, not a baseline!), and RETAIN (prior SOTA).
    - _Secret Terrible Thing_ There are some really obvious things to compare to here that are missing. What about just taking the output of the linear model `g_w` and passing that into the deep network `f_\theta` as an input (either in the timeseries, or at the output, etc.)? What about ensembling / multiple-instance-learning? What about residual learning? Similarly, is it the distillation or the `\zeta` training that is most helpful here? It seems `\zeta` in this case (the feature engineering associated with the linear model) involves a fair bit of domain knowledge.
    - 3 most relevant other papers:
      1) Knowledge distillation.
      2) Combination of baseline & neural models.
      3) RETAIN (prior SOTA).
    - Warrants deeper dive in main doc? (options: No, Not at present, Maybe, At least partially, Yes) Not really
  * **Detailed Methodology**:
    - Reverse Distillation:
      Reverse distillation is inspire by standard knowledge distillation paradigm. Let us take as given a binary prediction model 
      `f_\theta: \mathcal X \to \[0, 1\]` and a linear model `g_w: \mathcal X \to \[0, 1\]` defined by `g_w(x) = \sigma(w^T \zeta(x))` where `\sigma` is the sigmoid function, and `\zeta` is a fixed _feature engineering_ transformation `\zeta: \mathcal X \to \mathbb{R}^d` based on heuristic domain knowledge. Note that the linear model may outperform `f_\theta` for several reasons, including regularization, simplicity, and the quality of `\zeta` (this seems superficial to me -- surely `f_\theta` could also take as input `\zeta(x)`). Reverse distillation trains `f_\theta` over `\theta` so as to minimize the KL divergence between the bernoulli distributions introduced by `g_w` and `f_\theta`. At fine-tuning time, the model is trained via a weighted loss to both stay close to the linear model and to optimize direct predictive performance.
    - SARD: 
      SARD is a pretty standard transformer, with a deep set encoder used to represent a visit (which is a bag of codes) as the input embedding, a temporal embedding inspired by traditional sinusoidal positional embeddings for transformers, followed by a self-attention architecture, followed by a final convolution step and a max-pooling operation to consolidate to fixed-size.
  * **Key Strengths**:
    - Solves real problem
    - Nice ablation, baseline comparison (though RF also would've been valuable), and interpretability analyses
  * **Key Weaknesses**:
    - Lots of missing key comparisons.
    - No variances / significance testing.
    - Some ablations would be nice -- e.g., is the convolution necessary?
    - RD doesn't seem to add too much over SA w/o RD. 
  * **Open Questions**:
    - Is this amenable to generalization? From the context of the theory presented, I think probably, but it isn't presented as though it is, which I find confusing. Their theory says that the SARD model can perfectly mimic a linear model given perfect data, though how important that is I honestly don't know. This mostly just comes down to approximating the `\zeta`. 


## [Trajectory Inspection: A Method for Iterative Clinician-Driven Design of Reinforcement Learning Studies](https://arxiv.org/pdf/2010.04279.pdf)
  * **Logistics**:
    - Ji CX, Oberst M, Kanjilal S, Sontag D. Trajectory Inspection: A Method for Iterative Clinician-Driven Design of Reinforcement Learning Studies. arXiv preprint arXiv:2010.04279. 2020 Oct 8.
    - Time Range: 10/29/20 (09:58 - 10:01)
  * **Summary**:
    - _Single Big Problem/Question_ Treatment policies learned via reinforcement learning (RL) from observational health data are sensitive to subtle choices in study design.
    - _Solution Proposed/Answer Found_ The authors propose Trajectory Inspection, an iterative method to aid in study design for clinical RL applications. This approach finds predicted treatment trajectories that are especially aggressive and the corresponding modeled clinical trajectory accompanying those policies alongside the real course of the hospitalization to identify possible modeling issues.
    - _Experiments used to justify?_ They apply this approach to recent work on RL for inpatient sepsis management, identifying several issues related to study design.
  * **Key Strengths**:
    - Well written abstract, great idea, very useful in practice.
  * **Key Weaknesses**:
    - Need to read more to find out.
  * **Warrants further read**: Y/N

## [Neural Ordinary Differential Equations for Intervention Modeling](https://arxiv.org/pdf/2010.08304.pdf)
  * **Logistics**:
    - Gwak D, Sim G, Poli M, Massaroli S, Choo J, Choi E. Neural Ordinary Differential Equations for Intervention Modeling. arXiv preprint arXiv:2010.08304. 2020 Oct 16.
    - Korea University, KAIST, University of Tokyo
    - Time Range: 10/26/2020 09:24 - 09:58, 10/29/2020 09:47 - 09:54
  * **Summary**:
    - _Single Big Problem/Question_ Neural ODEs have emereged as a compelling framework for modeling irregular timeseries, but are not applicable to interventional data, which presents with both internal evolution of the timeseries as well as changing external inputs. 
    - _Solution Proposed/Answer Found_ The authors develope IMODE, a nove neural ODE method capable of working with interventional data by employing two ODE functions to separately handle observations and interventions.
    - _Why hasn't this been done before?_ Neural ODEs are still new, so this is directly building on existing research.
    - _Experiments used to justify?_
      1) Synthetic Experiments
      2) Real-world medical record datasets.
    - _Secret Terrible Thing_ Only real-world experiment is just on recontstruction of a single patient's trajectory. 
    - 3 most relevant other papers:
      1) [Neural Jump Stochastic Differential Equations](https://papers.nips.cc/paper/9177-neural-jump-stochastic-differential-equations.pdf)
    - Warrants deeper dive in main doc? Yes -- if nothing else, just to better understand the methodology here.
  * **Detailed Methodology**:
    So this is pretty complicated, and relies non-trivially on the formulation of the NJSDE framework. This system uses impulse differential equations, in which the system evolves under continuous dynamics for most of the time, (governed by some differential equation rules), but at discrete, irregular times, impulsive interventions impact the state in discrete jumps. Here, the authors have several distinct variables -- `h`, which is the overall "system state", `z_x`, which is the latent observational space, and `z_a`, which is the latent interventional (action, hence the "a") space. These evolve via continuous Neural DEs `f_{\psi, \theta, \phi}`, and jump DEs `g_{\theta, \phi}` -- why no `g_\psi`, you ask -- well, the assumption is that the impulse jump affects the latent spaces explicitly, but not the observed state itself, so dynamics in the observed space is always continuous, but the "velocity" so to speak varys in a discontinuous fashion. Note, however, that under this system, jumps affect both observation and intervention, which is interesting -- given that, why are they separate? Could we just do jumps on the interventional space?
    
    The system is trained on a reconstruction endpoint, going from `h` to the observed space `x`, only at observation points `x_{t_k}` (e.g., when the jumps are active). TODO: more methodological details
    
    Experimental Details
      - eICU. They just look a single patient with a long ICU stay history and only ICU stay interventions
  * **Key Strengths**:
    - This is minor, but it has a nice summary of existing attempts to use the Neural ODE framework for medical timeseries.
    - Nice synthetic dataset experiments.
  * **Key Weaknesses**:
    - I don't buy some of their "System-Specific Variants of IMODE" content. They seem to be trying to make first principles claims but aren't doing so with true proofs so it doesn't really seem correct to me.
    - Their experiment on the eICU dataset is just on a *single* patient! This is a bit crazy to me. They also only profile reconstruction error. 
  * **Open Questions**:
    - Under this framing, there is a clear and natural separation between observation points and non-observation points, which makes sense and seems to reflect reality. This begs the question of whether or the distribution `x_{t_k}` is different than that of `x_{\not t_k}` -- can we ever see the "observational space" at a _non_ observational time? This feels almost quantumy as a way to think about things.

## [Empirical Study of the Benefits of Overparameterization in Learning Latent Variable Models ](paper_link)
  * **Logistics**:
    - Buhai RD, Halpern Y, Kim Y, Risteski A, Sontag D. Empirical Study of the Benefits of Overparameterization in Learning Latent Variable Models. arXiv preprint arXiv:1907.00030. 2019.
    - 5 as of 10/05/2020
    - MIT, Harard, CMU, Google
    - Time Range: 10/05/2020 (15:34 - 16:00)
  * **Summary**:
    - _Single Big Problem/Question_ Understanding why overparametrization (training a very large model) is helpful without harming statistical performance, _in particular for unsupervised models_.
    - _Solution Proposed/Answer Found_ The authors perform an empirical study of different aspects of overparameterization in unsupervised latent variable models, finding that, like supervised models, overparametrization is widely helpful in this domain as well and has minimal harm. They also provide a concrete algorithm to discard false latent variables based on their prior probability and similarity to other discarded latent variables. Most interestingly, they _refute_ the possible explanation that overparametrization helps b/c within the large number of latent variables for an overparametrized model exist some that are very close to the true latent variables by tracking "matching" to latent variables over training.
    - _Why hasn't this been done before?_ Unclear.
    - _Experiments used to justify?_ A battery of experiments across network types and synthetic datasets.
      1) List of experiments used to justify (tasks, data, etc.) -- full context.
    - _Secret Terrible Thing_ What is the "secret terrible thing" of this paper?
    - 3 most relevant other papers:
      1) The supervised case: "[Zhang et al. (2016)](Understanding deep learning requires rethinking generalization)... showed that some neural network architectures that demonstrate strong performance on benchmark datasets are so massively overparameterized that they can “memorize" large image data sets (they can perfectly fit a completely random data set of the same size). Subsequent theoretical work provided mathematical explanations of some of these phenomena ([Allen-Zhu et al., 2018](http://papers.nips.cc/paper/8847-learning-and-generalization-in-overparameterized-neural-networks-going-beyond-two-layers); [Allen-Zhu & Li, 2019](http://papers.nips.cc/paper/9221-can-sgd-learn-recurrent-neural-networks-with-provable-generalization))."
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


## [The Lottery Ticket Hypothesis for Pre-trained BERT Networks](https://arxiv.org/pdf/2007.12223.pdf)
  * **Logistics**:
    - Chen T, Frankle J, Chang S, Liu S, Zhang Y, Wang Z, Carbin M. The Lottery Ticket Hypothesis for Pre-trained BERT Networks. arXiv preprint arXiv:2007.12223. 2020 Jul 23.
    - Texas A&M, MIT CSAIL
    - Time Range: 10/2/20 (16:14 - 16:27), 10/5/20 (13:35 - 13:59, 15:12 - 15:51)
  * **Summary**:
    - _Single Big Problem/Question_ Does BERT follow the lottery ticket hypothesis?
    - _Solution Proposed/Answer Found_ The findings here are nuanced. A couple of points:
      * They find true _winning tickets_ in BERT's pre-trained weights when examining downstream tasks, which hasn't been observed with deep CNNs or Transformers in general. This indicates that BERT's pre-training does some of the "dense initialization" discussed below.
      * In tasks without winning tickets, they still find matching subneteworks, albeit some that require a bit more dense training first to exist. This poses a natural question -- is there any correlation between the requirement to do dense pre-training and a failure to see pre-training improve performance most dramatically?
      * They find that winning tickets for MLM (and nearly only MLM) transfer universally to downstream tasks. For other tasks with large training sets, there is also some transferrability, though much less, and transferrability doesn't seem improved if using the initial pre-trained BERT weights rather than only partially rewound weights.
    - _Why hasn't this been done before?_ Lottery ticket is new, but this actually _has_ been done if not quite before, then at least concurrently, via [Prasanna et al's](https://arxiv.org/abs/2005.00561) study, that the authors contrast against that explicitly in this work.
    - _Experiments used to justify?_ Examines a vareity of lottery ticket pruning strategies on BERT fine-tuning tasks and a variety of relevant comparisons.
    - _Secret Terrible Thing_ No inductive take-aways, just observations. Other than that, no.
    - 3 most relevant other papers:
      1) "In larger-scale settings for computer vision and natural language processing, the lottery ticket methodology can only find matching subnetworks at an early point in training rather than at random initialization.... The phase of ptraining prior to this point can be seen as dense pre-training that creates an initialization amenable to sparsification. This pre-training can even occur using a self-supervised task rather than the supervised downstream task [19](https://openreview.net/forum?id=Hkl1iRNFwS)[20](https://openaccess.thecvf.com/content_CVPR_2020/html/Chen_Adversarial_Robustness_From_Self-Supervised_Pre-Training_to_Fine-Tuning_CVPR_2020_paper.html)." Read those two citations.
      2) "Finding matching subnetworks with the lottery ticket methodology is expensive... However, the resulting subnetworks transfer between related tasks [21](http://papers.nips.cc/paper/8739-one-ticket-to-win-them-all-generalizing-lottery-ticket-initializations-across-datasets-and-optimizers)[22](https://arxiv.org/abs/1905.07785)."
      3) "Although the lottery ticket hypothesis has been evaluated in the context of NLP and transformers [17](https://arxiv.org/abs/1906.02768)[18](https://arxiv.org/abs/2003.02389)[23](https://arxiv.org/abs/1902.09574), it remains poorly understood in the context of pre-trained BERT models... A concurrent study [24](https://arxiv.org/abs/2005.00561) also examines the lottery ticket hypothesis for BERTs."
    - Warrants deeper dive in main doc? Yes, time permitting.
  * **Detailed Methodology**:
    - Is mostly just a list of comparisons of fine-tuning results of various levels of IMP models.
  * **Key Strengths**:
    - Asks the right questions, in reasonable ways.
    - Findings are quite observationally interesting -- that MLM leads to best in class transfer performance in particular.
    - Could be impactful that IMP on MLM PT weights can lead to smaller BERTs that yield very similar performances.
  * **Key Weaknesses**:
    - Doesn't offer any inductive understanding -- I don't know how to predict based on these results what would happen were I to run iterative magnitude pruning on a protein model, for example.
    - Doesn't try to do any pruning during pre-training. That'd be the really powerful finding -- that one could do pruning during pre-training and still obtain excellent outputs.
  * **Open Questions**:
    - Is there any correlation between the requirement to do dense pre-training and a failure to see pre-training improve performance most dramatically?
  * **Extensions**:
    - IMP during pre-training
    
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
    - Warrants deeper dive in main doc? Not at present, but if I pursue a project in this space then yes.
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
 
## [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
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

# Latent Graph Learning
## [Differentiable Graph Module (DGM) for Graph Convolutional Networks](https://arxiv.org/pdf/2002.04999.pdf)
  * **Logistics**:
    - Kazi A, Cosmo L, Navab N, Bronstein M. Differentiable Graph Module (DGM) Graph Convolutional Networks. arXiv preprint arXiv:2002.04999. 2020 Feb 11.
    - Cited by 6 as of 10/19/20
    - Technical University of Munich, Germany, ICL, JHU, University of Lugano, Twitter.
    - Time Range: 10/19/2020 (13:54 - 14:02 \[not done yet\])
  * **Summary**:
    - _Single Big Problem/Question_ How can we do graph learning when the graph is dynamic or unknown?
    - _Solution Proposed/Answer Found_ Differentiable Graph Module, a learnable module predicting edge probability in the graph relevant for the task, and trained in an end to end fashion.
    - _Why hasn't this been done before?_ 1) GNNs are still somewhat new, 2) differentiable static operations are also new, 3) simultaneous learning and use of structure is a major challenge.
    - _Experiments used to justify?_ 
      1) Experiments across multiple domains
    - _Secret Terrible Thing_ Unknown
    - 3 most relevant other papers:
      1) DGCNN
      2) 
    - Warrants deeper dive in main doc? Yes
  * **Detailed Methodology**:
    - Graph edges are computed probabilistically based on euclidean distance between (featurized) nodes.
    - Graphs are then sampled using Gumbel-top-k trick (similar to what we do in adversarial protein pre-training work.
    - 
  * **Key Strengths**:
    - Strong results and methodology.
  * **Key Weaknesses**:
    - For segmentation task, compares to DGCNN even though that isn't SOTA on an of those categoreis, and the improvements observed here are not typically enough to punt it into SOTA territory.
  * **Open Questions/Extensions**:
    - Do you need to actually sample a graph in this setting? Or can you just use the probabilities wholesale via a GNN optimized for complete graphs?
    - Can we take advantage of the unstructured sparsity here somehow?

## [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/pdf/1801.07829.pdf)
  * **Logistics**:
    - Wang Y, Sun Y, Liu Z, Sarma SE, Bronstein MM, Solomon JM. Dynamic graph cnn for learning on point clouds. Acm Transactions On Graphics (tog). 2019 Oct 10;38(5):1-2.
    - Cited by 738 as of 10/19/2020.
    - MIT, UC Berkeley, ICL
    - Time Range: 10/19/2020 (13:26 - 13:52)
  * **Summary**:
    - _Single Big Problem/Question_ Processing point clouds is hindered by the lack of inherent topology in the space -- e.g., no graph connecting the points.
    - _Solution Proposed/Answer Found_ The authors propose using a dynamic graph that updates throughout the propagation of the network to add an inferred topology over the point cloud, which is then levergaed to improve processing of the point clouds.
    - _Why hasn't this been done before?_ Traditionally, graphs are assumed to be static and presented with data -- learning graphs from data is a harder task, and hasn't been connected to point cloud analyses previously, as point cloud modelling seems to be a relatively slow-moving field (most citations on this point are from 2015, 2017 or earlier).
    - _Experiments used to justify?_ 
      1) They perform analysis on several datasets and compare against a battery of benchmarks.
      3) They perform ablations to investigate the utility of their approach
    - _Secret Terrible Thing_ Unknown
    - 3 most relevant other papers:
      1) Non-local neural networks stuff
      2) Point cloud stuff
      3) learning graphs stuff
      4) GNN
    - Warrants deeper dive in main doc? Yes
  * **Detailed Methodology**:
    - graph is a kNN graph, where NN set is updated after each layer
  * **Key Strengths**:
    - Strong results, very intelligent approach.
  * **Key Weaknesses**:
    - Unclear how they're getting around this, but re-computing the NN graph after each layer of the Net must be _expensive_. Their model reports being more efficient than other approaches, so I don't understand this, but... No, wait, I think I follow. Each sample in their setting is a point _cloud_ not a point, so they are only re-doing the NN within a single sample, not across the whole dataset, and their individual samples must be small. Still, though, this raises the question of whether or not there aren't smarter ways to do this.
    - Their segmentation results don't seem that great? They don't comment on this much, though, so I wonder if I'm misunderstanding their evaluation. 
  * **Open Questions/Extensions**:
    - _How_ do they efficiently compute the pairwise distances? This doesn't appear to be answered in the main body.
    - Can graph be learned with classification rather than euclidean distance?
    - So this method looks like it is doing something like learning over instances, with a transformer aspect over all instances in the dataset to add in non-local information. Would a CLS analog help? Then are we also marrying neural turing machine / global RW block with learning over whole dataset?
    - Is there any relationship here to the PT perspective?
    - Is there any relationship here with interpolation based learning?

## [On the Bottleneck of Graph Neural Networks and its Practical Implications](https://arxiv.org/pdf/2006.05205.pdf)
  * **Logistics**:
    - Alon U, Yahav E. On the Bottleneck of Graph Neural Networks and its Practical Implications. arXiv preprint arXiv:2006.05205. 2020 Jun 9.
    - Technion
    - Time Range: 10/16/2020 (18:39 - 18:51)
  * **Summary**:
    - _Single Big Problem/Question_ Can graph neural networks truly effectively summarize information across arbitrarily sized graphs?
    - _Solution Proposed/Answer Found_ No -- there exists an inherent bottleneck in GNNs -- information across an exponentially growing set of nodes is "squashed" into a fixed-size embedding at the neighborhood aggregation step. This prevents the GNN from allowing information to flow from distant nodes and cripples the passage of long-range information. This is particularly true for GNNs that absorb information equally across edges. 
    - _Why hasn't this been done before?_ GNNs are relatively new? This is theoretical analysis.
    - _Experiments used to justify?_ 
      1) Controlled synthetic problem to demonstrate the existence of the problem
      2) Analytical & empirical analyses of GCN and GIN that demonstrate they are more sucsesptible.
      3) Show that "breaking" the bottleneck (by adding a fully connected graph layer at the top of their network, allowing full connection) improves performance on 3 real-world datasets.
    - _Secret Terrible Thing_ This same argument also applies to CNNs, but doesn't seem to be a problem there. Why is that?
    - 3 most relevant other papers:
      1) Classic GNN papers
      2)
      3)
    - Warrants deeper dive in main doc? Yes
  * **Detailed Methodology**:
    
  * **Key Strengths**:
    - List of big pros of the paper
  * **Key Weaknesses**:
    - No theoretical analysis proving existence of bottleneck
    - I'm not totally convinced by their theoretical examples -- of course you need to increase the embedding size with larger graphs, but they don't. 
  * **Open Questions/Extensions**:
    - My graph transformer idea!
    
## [Grale: Designing Networks for Graph Learning](https://arxiv.org/pdf/2007.12002.pdf)
  * **Logistics**:
    - Halcrow J, Mosoi A, Ruth S, Perozzi B. Grale: Designing Networks for Graph Learning. InProceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining 2020 Aug 23 (pp. 2523-2532).
    - Google Research, YouTube
    - Time Range: 10/16/2020 (18:28 - 18:38)
  * **Summary**:
    - _Single Big Problem/Question_ How should one construct a high-quality, sparse graph optimal for semi-supervised learning given a collection of weaker, more dense graphs?
    - _Solution Proposed/Answer Found_ Grale -- a scalable method for graph design which fuses together different measures of potentially weak similarity to create a graph which exhibits high task-specific homophily.
    - _Why hasn't this been done before?_ GNNs are relatively new, and this problem is one that will only present at scale and in industrial applications -- in research, we can design alternate models or pick other settings that don't have the weak graph problem.
    - _Experiments used to justify?_
      1) Case study to use Grale to flag abuse classification problems on YouTube over 100s of Millions of items, in which Grale (by enabling a semi-supervised approach atop rule-based and content-based classifiers) improves recall by 89%.
      2) Experiments on two other datasets: USPS & MNIST (WDTMT?).
    - _Secret Terrible Thing_ This only applies to a single task of interest -- in many contexts, we'd like to use a single graph for many tasks, rather than having to re-design the graph each time. Pending scalability concerns, it may not even be feasible to build a copy of the graph each time.
    - 3 most relevant other papers:
      1)
      2)
      3)
    - Warrants deeper dive in main doc? Not at present -- skipping rest of paper.
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


# QA
## [What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams](https://arxiv.org/pdf/2009.13081.pdf)
  * **Logistics**:
    - Jin D, Pan E, Oufattole N, Weng WH, Fang H, Szolovits P. What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams. arXiv preprint arXiv:2009.13081. 2020 Sep 28.
    - Time Range: 10/15/20 (13:41 - 13:51)
  * **Summary**:
    - _Single Big Problem/Question_ Lack of medical-domain multiple-choice QA datasets
    - _Solution Proposed/Answer Found_ Build MedQA (3 versions; English, Traditional Chinese, Simplified Chinese)
    - _Experiments used to justify?_ Baselines 
  * **Key Strengths**:
    - Strong dataset, with well defined questions and answers, sourced from medical board exams in US, Mainland China, and Taiwan. 
    - Evidence base collected & documented, with assessment of viability.
    - Nice analysis of failure cases
  * **Key Weaknesses**:
    - Limited relevance to emrQA (dataset is multiple choice, not free-text) and no methodological novelty (which is intentional, as this is a QA dataset paper).
  * **Warrants further read**: Maybe, pending EMRQA project.

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
    - UCLA, MSR
    - Time Range: 10/1/20 (13:24 - 13:34), 10/22 (09:18 - 09:50)
  * **Summary**:
    - _Single Big Problem/Question_ Pre-training over graphs for node-classification, link prediction, or subgraph classification (though they only empirically analyze node classification).
    - _Solution Proposed/Answer Found_ Hu et al propose a generative pre-training objective based on masked graph generation, masking both nodes and edges.
    - _Why hasn't this been done before?_ Why has nobody solved this problem in this way before? What hole in the literature does this paper fill? (1 sent).
    - _Secret Terrible Thing_ What is the "secret terrible thing" of this paper?
    - _Experiments used to justify?_ Comparison across the open academic grpah and the amazon graph for node classification, comparing to graph autoencoder, graph sage, and graph infomax. They also show that the HGT offers strong performance as the base model here. TODO: details.
    - 3 most relevant other papers: TODO
    - Warrants deeper dive in main doc? Y, and in addition the [Heterogenous Graph Transformer](https://dl.acm.org/doi/abs/10.1145/3366423.3380027?casa_token=neLIzbBgs70AAAAA:9PVQ1y_5p06rxyI28--hR5D6dGFH2e9_FBEJoxh_SxwJCYMOCiRHjil_lU8tyFDY3klfyx15OUuzsA) paper also warrants a look.
  * **Detailed Methodology**:
    - First step is to model likelihood over graphs (`p(G;\theta)`). How to do this? Most methods assume an _order_ to the nodes (albeit one that is arbitrary). Then, autoregressive factorizations can be used -- e.g., the probability of the graph is the probability of the first node multiplied by the probability of the rest of the graph conditioned on the first node and its outgoing edges. They do the same here (eq. 2).
    - How do you actually model the probability of the new node given the rest of the graph? Well, you can do this in an autoregressive manner. E.g., `p(X_i, E_i | X_{<i}, E_{<i}) = p(X_i | ...) p(E_i | X_i, ...)`, and you can further do the same for the edges in `E_i` in an arbitrary order --- Note that in practice, however, in this work, they assume each edge generated is independent of every other edge generated.
    - How do they actually do this, technically? The authors use a standard decode architecture to impute their masked nodes, and a pretty standard setup for edge generation as well. One note of something they do differently is separate each masked node into an "Attribute Generation Node" and an "Edge Generation Node". They do this such that both nodes have identical edge structures, and the attribute generation node also has an additional outgoing edge to the edge generation node. The attribute generation node is initialized with a `[MASK]` embedding and the edge generation node with the true node embedding. 
  * **Key Strengths**:
    - Strong experimental results, reasonable idea.
  * **Key Weaknesses**:
    - Only analyzes node classification, in practice.
    - Their is a pathway for attribute leakage in this system, via the edge generation nodes and retained observed edges for the node. This violates their theoretical guarantees, I think, at least for attribute generation.
    - Their system is using teacher forcing, or something like it, as the edge generation is always based on the true node embeddings. This is fine for the purposes of their pre-training strategy, but if they really wanted to use this system to generate graphs wholesale, it would be problematic.
  * **Open Questions**:
    - List of open questions inspired by this paper
  * **Extensions**:
    - List of possible extensions to this paper, at any level of thought-out.
  * **How to learn more**:
    - List of terms/concepts/questions to investigate to learn more about this paper.

## [Heterogeneous Graph Transformer](https://dl.acm.org/doi/abs/10.1145/3366423.3380027?casa_token=neLIzbBgs70AAAAA:9PVQ1y_5p06rxyI28--hR5D6dGFH2e9_FBEJoxh_SxwJCYMOCiRHjil_lU8tyFDY3klfyx15OUuzsA)
  * **Logistics**:
    - Hu Z, Dong Y, Wang K, Sun Y. Heterogeneous graph transformer. InProceedings of The Web Conference 2020 2020 Apr 20 (pp. 2704-2710).
    - 28 citations (10/22/2020)
    - UCLA, MSR
    - Time Range: 10/22/20 (09:52 - 09:59)
  * **Summary**:
    - _Single Big Problem/Question_ How to model heterogeneous graphs?
    - _Solution Proposed/Answer Found_ Heterogenous Graph Transformer (HGT) architecture has specialized attention mechanisms for disparate node- and edge- types. Instead of using parametrized edge types, this system also incorporates node types of the endpoints of the edge, and these are used to actually parametrize the weight matrices directly for calculating attention over each edge, thus allowing nodes and edges of different types to maintain separate representation spaces. (why is this desirable)?
    - _Why hasn't this been done before?_ GNNs are under active development now, but large scale systems, methods, and data are new. Plus, transformers are still fresh.
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
