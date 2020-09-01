# Research Paper Quick Hits
This doc contains brief notes on _skimmed_ papers (no more than 20 min) per paper, which ultimately becomes a feeder into the more in depth notes in `README.md`. 

### Acronyms
  1. WDTMT?: "What does this mean, technically?"
  2. ITT?: "Is this true?"
  
### Expected Format

```
# [Paper_Title](paper_link)
  * **Logistics**:
    - Citation (key points: publication venue, date, authors)
    - \# of citations of this work & as-of date
    - Author institutions
    - Time Range: START - END
  * **Summary**:
    - _Single Big Problem/Question_ The single big problem the paper seeks to solve (1 sent).
    - _Solution Proposed_ The proposed solution (1 sent).
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
  * **Pros**:
    - List of big pros of the paper
  * **Cons**:
    - List of big cons of this paper
  * **Open Questions**:
    - List of open questions inspired by this paper
  * **Extensions**:
    - List of possible extensions to this paper, at any level of thought-out.
  * **How to learn more**:
    - List of terms/concepts/questions to investigate to learn more about this paper.
```

# [Hierarchical Attention Propagation for Healthcare Representation Learning](https://dl.acm.org/doi/abs/10.1145/3394486.3403067)
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

# [HiTANet: Hierarchical Time-Aware Attention Networks for Risk Prediction on Electronic Health Records](https://dl.acm.org/doi/abs/10.1145/3394486.3403107)
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
