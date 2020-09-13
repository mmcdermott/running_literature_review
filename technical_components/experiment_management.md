# Experiment Management Solution
Technical System Learning. Began 08/23, Finish TBD

This is just a placeholder to flag this resource: http://akosiorek.github.io/ml/2018/11/28/forge.html. See also the many other recommendations in the first comment.

## PyTorch Lightning, 09/13/2020
Today, I'll walk through [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning). Let's start by compiling a list of relevant resources publicly available:
  * PyTorch Lightning Itself: [Github](https://github.com/PyTorchLightning/pytorch-lightning#how-to-use), [Docs](https://pytorch-lightning.readthedocs.io/en/latest/)
  * https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
  * https://www.youtube.com/watch?v=QHww1JH7IDU
  
Importantly, that towardsdatascience link gives us this Colab notebook, which we can work through directly: https://colab.research.google.com/drive/1Mowb4NzWlRCxzAFjOIJqUmmk_wAT-XP3
### Notes on https://colab.research.google.com/drive/1Mowb4NzWlRCxzAFjOIJqUmmk_wAT-XP3
  1. The restoration cell isn't designed to work out of the box.
  2. A couple of other pieces are similarly not designed to work end to end, and are more evocative. Additionally, this is a static tutorial -- no parts for me to edit, and therefore minimal reason to do more than read the raw code. Instead, I'll just take some more general notes.
  
#### Changes from raw PyTorch
Uncategorized list:
  1. In Lightning, forward is just the representer -- the loss part happens in the train_loop part.

PyTorch Lightning builds on the style of the "master/monster model/module". The Lightning module includes both the `forward` method of a traditional PyTorch module, but additionally a number of other specialty methods:
  1. [`self.training_step`](https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html#training-step)
     This method defines the "training step" part of traditional PyTorch code. It can return either (maybe?) a flat dict or a [Lightning TrainingResult](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.step_result.html#pytorch_lightning.core.step_result.TrainResult). This function allows you to simply compute your loss and not worry about doing anything more -- of course, the flipside to this is that if you do something more complicated than just a typical loss, you may get bit :/
     They give some examples on how they try to combat this potential problem:
       a) GANs and other simultaneous optimization problems are handled via an extra (positional) `optimizer_idx` argument. This seems a bit clunky to me -- I like things to be named, not defined (solely) by index. Additionally, the positional nature of this is not great.
       b) Truncated back-propagation through time is also handled by an extra (again positional) `hiddens` argument. 
       c) Coordination across multiple GPUs in the case of distributed training also needs to be handled separately. In my opinion, the way they handle this is the cleanest of the three ways they handle things in these edge cases -- there is a function `training_step_end` which you use to run the coordination steps, which receives as an input the `TrainingResult`s of the various single-GPU parts.
  2. [`self.configure_optimizer`](https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html#configure-optimizers)
     This method configures optimizers for the learning process. As with training, you need to play some games to account for, for example, multiple optimizers. Note that this method also entails configuration of the learning rate schedulers. 
  3. The source colab points to some nested data prep functions, but I don't see those documented... I think instead those have been abosrbed by the [LightningDataModule](https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html). Ok, hmm. Another colab, linked from the source page itself, also has them integrated. I think this is an optional/ambiguous API situation. 
     Found it! These are examples of [_Data Hooks_](https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html#data-hooks), where a lightning module can be directly coupled to a dataset.
     Now that that's resolved, some notes on `LightningDataModule`s:
       a) In `prepare_data()`, _don't_ set state on `self`.
       b) You can manually run `prepare_data()` and `setup()` independently of Lightning, in case your model needs access to a sample data element to run (for example): https://pytorch-lightning.readthedocs.io/en/stable/datamodules.html#using-a-datamodule

PyTorch Lightning also has some nice convenience methods:
  1. https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html#save-hyperparameters
     This can work especially nicely with my args system setup --- see the single_arg case.
     
[Hooks](https://pytorch-lightning.readthedocs.io/en/stable/lightning-module.html#hooks) also worth reviewing. 

### Questions:
How will this system fare with
  1) Models that pass dictionarys around like candy?
  2) Highly extensible systems, where you basically do architecture search via hyperparameter tuning?
  3) Multi-optimizer systems, like GANs?
  4) My `args` package & one-dir-one-run paradigm.
    a) They have [some feedback](https://pytorch-lightning.readthedocs.io/en/latest/hyperparameters.html) on using this in concert with `argparse`
    b) They also link to an interesting set of utilities I'd not heard of before, called `hydra`: https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710. It deserves some commentary in the `args` package.
  5) Dataset caching/checkpointing?
  6) Meta-learning systems?
  7) Auto-hyperparameter tuning systems, e.g., Optuna/Hyperopt/Ax-Botorch?
    a) [Lightning + Optuna](https://medium.com/pytorch/using-optuna-to-optimize-pytorch-lightning-hyperparameters-ce1cc1a034a)
       This seems like it works pretty well. Optuna even has a [specific callback integration](https://optuna.readthedocs.io/en/latest/reference/integration.html#optuna.integration.PyTorchLightningPruningCallback) for Lightning to support early pruning of trials. An example of this in action is [here](https://github.com/optuna/optuna/blob/master/examples/pytorch_lightning_simple.py).
