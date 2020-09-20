# Argument Passing
Technical Learning System Exploration:
  * 08/23/2020 (background research, starting porting code over to separate repo)
  * 08/30/2020 (finalized v0 of repo; added tests, validated code, and ensured installability)
  * 09/13/2020 (Added commentary on Hydra)

## Motivation
Argument parsing is something that is done in some manner for every project. However simple it appears, however, it can be both (1) more complicated, and (2) have implications on the broader project's API and suitability to best-practices than we expect. Some examples:
  1. Arguments are your projects main "interface." A poor interface can make it harder to test, use, extend, document, and understand.
  2. Arguments allow you to (in effect) add a layer of typing/validation to your project, that can help catch immediate errors and keep your project working properly. This happens at several levels:
    * Hard validations: Validations that catch errors from the outset, before your system spends any time on setup that would be wasted by an error that would occur later (e.g., submitting the string "0" to your dropout arg, rather than the float 0).
    * Soft validations: Validations that catch things that won't necessarily cause errors immediately, but likely (should give warning) or certainly (should error) don't make sense (e.g., passing a dropout probability of 1.0, which doesn't make sense as everything would thus be dropped out, though your model could likely still run without throwing an immediate error in such a setting).
  3. Setting your arguments up in a manner that they are _always_ written out to a file for permanent recording in your experimental directory helps keep all your code much more reproducible. Similarly, setting up your system to be able to run from arguments passed by file helps make it easier to work with your system when the spec grows more complex without relying on large, brittle command line strings.
  4. We commonly run the same system through multiple endpoints (e.g., Jupyter Notebooks, Slurm, and Command Line directly). Having a canonical interface for argument passing that's "above" argparse is valuable here b/c it gives us a way to use the same argument processing system and interface across endpoints.

These facts make argument parsing worth careful consideration. In addition, given how central argument parsing is to so many other kinds of CS projects (and thus how likely it is that others smarter than I have put careful thought into this) suggest that there are likely great best practices we can draw on here, which also makes it a compelling target for more careful analysis. 

There's another aspect to argument processing that is also useful -- namely, working effectively across multiple setups / machines. Ostensibly, we'd like a project to operate effectively in the context of an "Environment Configuration", be that encoded by hand, or, preferrably, in a file somehow. This can be approximated through the use of environment variables (e.g., `PROJECT_ROOT_DIR`) or something, which points to a general directory with subdirs listed according to project names, which are specified in the project via a constant, but occassionally, for whatever reason, you need to support a more complicated overriding of operational constants in a system dependent manner. This would also be nice to handle effectively.

Lastly, in ML specifically, a common issue in argument management / configuration is that of _composability_. If I'm building a model to process clinical timeseries, for example, My model may need to have several pieces, including (1) an embedding layer which casts non-continuous inputs into a continuous embedding layer, (2) a time-varying input encoder, which reduces the input timeseries to a common embedding space, and (3) a task decoder which decodes from the fixed size embedding layer into the output space (e.g., making a prediction). For each layer, I could choose to use different kinds of models, with different sets of hyperparameters--most notably, for (2), I could choose to use a GRU, a CNN, a transformer, an aggregation + linear model, etc. The full "hyperparameter space" of the system thus includes a discrete "model choice" parameter which flags which model type I want to use, followed by the _nested_ parameters for that model specifically (e.g., # and size of layers, uni- or bi-directional (for GRU), kernel size (for CNN), and # of attention heads (for transformer)). How we handle these composable arguments is also important and could likely be handled in intelligent manners.

I'll start this exploration by going through some existing resources I've found (current as of 08/2020), then progress to broader commentary on any remaining holds, recommendations for best practice. If any of my existing code for this has a place here, then I'll also spend some time pulling that out into a separate repo/python app and can describe that here.

## Literature Review / Existing Systems & Best Practices
### Canonical Argument Parsing Libraries:
  * [`argparse`](https://docs.python.org/3/library/argparse.html#module-argparse). This is the canonical argument parsing solution for Python. I'll presume readers are at least roughly familiar with it. If not, you should read the link above first to get used to the library.
  * [`--docopt`](http://docopt.org/). `docopt` lets you define, rather than an immediate interface, a "Usage String," which serves both as a documentation, and by standardizing canonical best practices for such strings in unix systems, a definition of arguments.
  * [Click](https://click.palletsprojects.com/en/5.x/) Seems to be designed more with power in mind, rather than immediate usability (`argparse`) or the "magic" of going straing from a help string (`--docopt`). Click uses decorators, let's you decorate a function as a command, add arguments, etc. It is desigend to work across python 2 and 3, and [the authors have put definite thought into why this is necessary](https://click.palletsprojects.com/en/7.x/python3/), support things like colors in print output, easy integration with setuptools, typing, etc. Click seems like it would possibly be best for [large, complex applications](https://click.palletsprojects.com/en/7.x/complex/). I doubt most ML pipelines apply. Click also supports more advanced features, like [testing](https://click.palletsprojects.com/en/7.x/testing/), [terminal interaction](https://click.palletsprojects.com/en/7.x/utils/), [command line completion (if integrated through setuptools)](https://click.palletsprojects.com/en/7.x/bashcomplete/)
  * [Abseil](https://abseil.io/docs/python/quickstart) Google has their own utility, in Abseil. This is a bit different than other systems, in that it has a _distributed_ model for flags. Rather than, for example, you needing to define a single argument system in your "main" you define FLAGS for the interface of each part separately in their unique files. Then the global spec is combined based on python imports. Note that this speaks to the "composability" issue we identified above, at least in part. It also supports flag validators. It does, however, seem to have poor documentation and I don't know how effectively it would merge with Jupyter.
  * (09/13/20 Entry) [Hydra](https://medium.com/pytorch/hydra-a-fresh-look-at-configuration-for-machine-learning-projects-50583186b710). At first glance, Hydra seems tailor made to this situation. I'm going to check it out in more detail and leave commentary below.
  
There is also [this blog post](https://realpython.com/comparing-python-command-line-parsing-libraries-argparse-docopt-click/) which serves to compare the three libraries. Additionally, [Click has a comparison page](https://click.palletsprojects.com/en/5.x/why/) as well.

### Jupyter Notebooks
  * [Blog post](https://www.dataquest.io/blog/advanced-jupyter-notebooks-tutorial/), [underlying library (papermill)](https://github.com/nteract/papermill)
    Papermill provides a way to parametrize Jupyter notebooks and then control those parameters when executing the jupyter notebook on the command line (e.g., as a script). This could be further useful for result visualization and analysis, when we may want to, for example, predominately have the code stored in a jupyter notebook as interactivity is critical, but also be able to run it via the command line for different experiments when nothing has changed. It would be nice (but non-essential) if canonical argument structures could also speak to papermill.

### Deep Learning Specific Resources:
I'll focus on 2 things here:
  1) General recommendations for configuration management, and 
  2) Utilities of existing libraries (e.g., PyTorch and Tensorflow) for this problem.
  
#### General recommenations for configuration management
  1. Use config files. This is repeated in a number of places. It also agrees with some of the points I raised in the motivation section above. The most trustworthy seeming source stating such is [this excerpt from Stanford CS230](https://cs230.stanford.edu/blog/hyperparameters/). They also comment more generally on the hyperparameter search aspect of this and connection to this to storage of parameters.

#### Utilites of existing libraries
  1) [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning). This doesn't seem to do anything towards argument parsing, but has great support for other aspect of improving runnability of ML pipelines. Worth a tutorial session in the future.
  2) I'm not seeing anything within Tensorflow that solves this directly.
  3) Ditto for Keras, or Keras-tuner.

## My Code
Based on this, admittedly preliminary literature review, I think there is still a use case for my code, at least for now. I still want to be able to easily translate from command line args to jupyter notebook arguments, and I want to be able to trivially write and read args from files. Later functionality, like (1) improved validation (again, regardless of medium), (2) better composability for nested models, and (3) support for more complicated argument types (e.g., file objects) could also be nice.

My utility is tracked here: https://github.com/mmcdermott/multisource_args. Currently, though it requires you to specify the argument parsing spec twice (once as a dataclass, once in argparse), it does have nice utilities for writing to and reading from files, both in abstract and from the commandline. The utility is tested, can be installed via pip (from the github), and plans are in place (but will require additional work) to streamline the interface by eliminating the duplicated spec issue.

## Hydra
Hydra is a new system (v1 on Sep. 3, 2020) for dynamic, nestable configuration files. At first blush, I think it is likely _the_ right solution for what my code is intended to do. It builds on [OmegaConf](https://omegaconf.readthedocs.io/en/latest/index.html) which is something I also should've found in my initial search. I think the likely failure in search here was that I was primarily thinking of this as a "command line/argument management system", because that's how I ended up here, but given how appropriately file-centric the final system is, I should've searched for things in the reverse direction, as a "config file management system". Hydra supports some nice features, inclduing:
  * Native support for sweeping via Ax: https://hydra.cc/docs/plugins/ax_sweeper/
  * Overridding default configuration files: https://hydra.cc/docs/advanced/hydra-command-line-flags/#--config-path-cp
    - Bonus, _having_ default configuration files! That's a winner.
  * Native support for Jupyter notebooks: https://hydra.cc/docs/advanced/jupyter_notebooks/
    So, this is underwhelming. Firstly, the notebook they link to doesn't actually exist, but when you navigate manually to the _actual_ notebook, it doesn't really do what I'm looking for. That said, I think you can just build the config namespace object yourself and you'll be fine as you would in any other case. Or, similarly, you can also use a structured configuration object and instatinate the config class yourself.
  * Native support for unit testing Hydra-driven applications: https://hydra.cc/docs/advanced/unit_testing/
  * Native support for slurm launching: https://hydra.cc/docs/plugins/submitit_launcher/
  * There's commentary in the community for how to integrate this with things like PyTorch Lightning: https://towardsdatascience.com/keeping-up-with-pytorch-lightning-and-hydra-31e8ed70b2cc
  * Structured configs, are, I think, the full answer to replace my package. They let you use dataclasses to define configurations, and you can associate validation as well.
  
I have one question left -- can you combine structured configurations and user-specified overwrite flat config files? [Here](https://hydra.cc/docs/tutorials/structured_config/intro) is where to look to find the answers.

See https://github.com/mmcdermott/test_project_hydra_lightning_ax for a test project using Hydra
