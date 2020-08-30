# Hyperparameter Tuning
## Work Log:
  * 08/30/2020 - began speccing out this investigation (only 15 minutes allocated)

## Goal
Hyperparameter tuning is essential. This is especially true for Deep Learning systems in comparison with other baseline methods. There are two realms of concern of interest to me in this investigation. The former is more technical, and will be the focus on the practical component of this investigation. The latter is more theoretical in nature. I detail both of these in detail below. At the end of this doc I'll also put some comments in about hyperparameter tuning in other contexts (e.g., non deep learning; scikit-learn, for example).

### Concern 1: What is the best system out there for hyperparameter tuning, and how to best use it?
There are a number of systems one could use for hyperparameter tuning. A current list of options (unordered)
  * [Hyperopt](http://hyperopt.github.io/hyperopt/)
  * [RayTune](https://docs.ray.io/en/latest/tune/index.html) (specific comparison [note](https://docs.ray.io/en/latest/tune/index.html#why-choose-tune))
  * Optuna, [website](https://optuna.org/), or [paper](https://arxiv.org/pdf/1907.10902.pdf)
  * [Botorch/Ax](https://ax.dev/tutorials/tune_cnn.html)
  * [AutoPytorch](https://www.automl.org/automl/autopytorch/)
  
Meta-lists:
  * [Optuna vs. Hyperopt](https://towardsdatascience.com/optuna-vs-hyperopt-which-hyperparameter-optimization-library-should-you-choose-ed8564618151)
  * Just some big list: https://github.com/balavenkatesh3322/hyperparameter_tuning
  * Check the page in the list above for Tune's comparison point.

Cloud compute specific systems:
  * [GCP](https://cloud.google.com/ai-platform/training/docs/hyperparameter-tuning-overview)

### Concern 2: What are best practices for hyperparameter tuning in general, especiall w.r.t. model comparison, speed of search, etc.?
Some example questions here:
  * How can we meaningfully ensure we're giving appropriately equal / fair computational power to methods of different complexities in model comparisons?
    - e.g., Does it make sense to work at the level of # of samples? # of computational cycles required? amount of time? density of parameter space explored?
  * How can/should we adjust for multiple comparisons when hyperparameter tuning at different scales across methods?
    - e.g., If I do `N` samples for hyperparameter search for baseline and `M` for model, should I do FDR correction at level `N*M`?
  * How can we use "epoch" / "training iteration" as a sentinel notion in hyperparameter tuning deep models?
    - e.g., if the model sucks at epoch 5, maybe we don't need to run it to epoch 10?
    
## Notes
Largely, these notes will focus on the empirical questions -- how do the various systems differ, which one seems to work best, notes on working through any tutorials, etc.

### Empirical Comparison of Frameworks
A central point here is to think about what the goals/requirements are of the system. For example, a few considerations:
  * ML Library (e.g., PyTorch, Tensorflow, general, etc.)
  * Distributed?
  * 

### Brief notes on theoretical questions.
Broadly, these questions speak to two areas: model comparison & automated hyperparameter tuning. Some possible good resources (will be) listed below. These may get promoted for lit review analysis

#### Model Comparison
#### Automated Hyperparameter Tuning
   
### Hyperparameter tuning for non deep-learning
Scikit-learn also has some 
