# Evaluation and calibration with uncertain ground truth

This repository contains code for our papers on calibrating [1] and evaluating [2] AI models with uncertain and ambiguous ground truth.

Currently, the code allows to reproduce our results on the toy dataset
presented in [1]. *Results on the dermatology case study of [2] will
follow soon.*

```
[1] Stutz, D., Roy, A.G., Matejovicova, T., Strachan, P., Cemgil, A.T.,
    & Doucet, A. (2023).
    [Conformal prediction under ambiguous ground truth](https://openreview.net/forum?id=CAd6V2qXxc).
    TMLR.
[2] Stutz, D., Cemgil, A.T., Roy, A.G., Matejovicova, T., Barsbey, M., Strachan,
   P., Schaekermann, M., Freyberg, J.V., Rikhye, R.V., Freeman, B., Matos, J.P.,
   Telang, U., Webster, D.R., Liu, Y., Corrado, G.S., Matias, Y., Kohli, P.,
   Liu, Y., Doucet, A., & Karthikesalingam, A. (2023).
   [Evaluating AI systems under uncertain ground truth: a case study in dermatology](https://arxiv.org/abs/2307.02191).
   ArXiv, abs/2307.02191.
```

![Monte Carlo conformal prediction teaser](https://davidstutz.de/wordpress/wp-content/uploads/2023/11/teaser-monte-carlo-conformal-prediction.jpg)

![Evaluation teaser](https://davidstutz.de/wordpress/wp-content/uploads/2023/11/teaser-uncertain-ground-truth.jpg)

## Overview

For safety, AI systems often undergo thorough evaluation and targeted calibration against a ground truth that is assumed certain. However, in many cases, this is actually not the case and the ground truth may be uncertain. Unfortunately, this is largely ignored in practice even though it can have severe consequences such as overestimating of future performance and mis-calibration. To address this problem, we present work for taking uncertain ground truth into account when evaluating and calibrating AI models.

For evaluation, we assume that ground truth uncertainty decomposes into two main components: annotation uncertainty which stems from the lack of reliable annotations, and inherent uncertainty due to limited observational information. This uncertainty is ignored when estimating the ground truth by deterministically aggregating annotations, e.g., by majority voting or averaging. In contrast, we propose a framework where aggregation is done using a statistical model. Specifically, we frame aggregation of annotations as posterior inference of so-called plausibilities, representing distributions over classes in a classification setting, subject to a hyper-parameter encoding annotator reliability. Based on this model, we propose a metric for measuring annotation uncertainty and provide uncertainty-adjusted metrics for performance evaluation.

For calibration, we use conformal prediction (CP) which allows to perform rigorous uncertainty quantification by constructing a prediction set guaranteeing that the true label is included with high, user-chosen probability. However, this framework typically assumes access to certain labels on a held-out calibration set. Applied to labels obtained through simple majority voting of annotations, the obtained coverage guarantee has to be understood w.r.t. these voted labels -- not the underlying, unknown true labels. Especially if annotators disagree strongly, the distribution of voted labels ignores this uncertainty. Therefore, we propose to directly leverage the annotations to perform CP. Specifically, we use plausibilities obtained from the above statistical aggregation to sample multiple pseudo-labels per calibration examples. This leads to Monte Carlo CP which provides a coverage guarantee w.r.t. the obtained plausibilities rather than the voted labels.

In a case study of skin condition classification with significant disagreement among expert annotators, we show that standard deterministic aggregation of annotations, called inverse rank normalization (IRN) ignores any ground truth uncertainty. We develop two alternative statistical aggregation models showing that IRN-based evaluation severely over-estimates performance without providing uncertainty estimates. Moreover, we show that standard CP w.r.t. to the voted labels obtained from IRN under-covers the expert annotations while our Monte Carlo CP closes this gap.

## Installation

1. Install [Conda](https://docs.conda.io/en/latest/) following the
   official instructions. Make sure to restart bash after installation.
2. Clone this repository using

    git clone https://github.com/deepmind/git
    cd uncertain_ground_truth

3. Create a new Conda environment from `environment.yml` and activate it
   (the environment can be deactivated any time using `conda deactivate`):

    ```
    conda env create -f environment.yml
    conda activate uncertain_ground_truth
    ```

   Alternatively, manually create the environment and install the following
   packages:

    ```
    conda create --name uncertain_ground_truth
    conda activate uncertain_ground_truth
    # TensorFlow only required for colab_mnist_multi_label.ipynb, but if wanted we
    # recommend installing it first.
    conda install -c conda-forge tensorflow
    conda install -c conda-forge tensorflow-datasets
    conda install -c conda-forge absl-py scikit-learn jax
    conda install jupyter matplotlib
    ```

4. Check if all tests run:

    ```
    python -m unittest discover -s . -p '*_test.py'
    ```

   Make sure to always start jupyter from within the
   `uncertain_ground_truth` environment. Then the preferred kernel selected
   by Jupyter will be the correct kernel.

These instructions have been tested with Conda version 23.7.4 (not miniconda)
on a 64-bit Linux workstation. We recommend to make sure that no conflicting
`pyenv` environments are activated or `PATH` is explicitly set or changed in
the used bash profile. After activating the Conda environment, the corresponding
Python binary should be first in `PATH`. If that is not the case (e.g.,
`PATH` lists a local Python installation in `~/.local/` first), this can
cause problems.

## Usage

All of this repository's components can be used in a standalone fashion.
This will likely be most interesting for the standard conformal prediction
(`conformal_prediction.py`), Monte Carlo conformal prediction
(`monte_carlo.py`), p-value combination (`p_value_combination.py`), and
plausibility regions (`plausibility_regions.py`) methods from [1]. Note that the
plausibility regions have been removed for clarity in the TMLR version of [1]
but is available in [v1 on ArXiv](https://arxiv.org/abs/2307.09302v1).
Moreover, this repository includes implementations of the Plackett-Luce Gibbs
sampler (`pl_samplers.py`), probabilistic IRN (`irn.py`),
partial average overlap (`ranking_metrics.py`) and general top-k
(aggregated) coverage and accuracy metrics (`classification_metrics.py`).
This will likely be most interesting regarding the (smooth) conformal prediction
Finally, the toy example from [1] can be found in `gaussian_toy_dataset.py`.

### Reproducing experiments from [1]

1. Start by running `colab_toy_data.ipynb` to create the toy dataset
   used for illustrations throughout [1]. The Colab will visualize the dataset,
   save it to a `.pkl` file and train a few small MLPs on the dataset.
2. Run `colab_toy_mccp` to re-create many of the plots from Sections 2 and 3.
   The Colab includes examples of running standard conformal prediction against
   voted labels, Monte Carlo conformal prediction against plausibilities
   (with and without ECDF correction), and plausibility regions against
   plausibilities.
3. Run `colab_mnist_multi_label.ipynb` to run Monte Carlo conformal prediction
   on a synthetic multi-label dataset derived from MNIST.

### Reproducing experiments from [2]

1. Start by running `colab_toy_data.ipynb` to create the toy dataset from [1].
   This will also sample multiple annotations as (partial) rankings per example.
2. Run `colab_toy_pl_sampler.ipynb` to run the Plackett-Luce Gibbs sampler on
   the sampled annotations from 1.
3. Run `colab_toy_ua_accuracy.ipynb` to re-create figures from [2] on the toy
   dataset using probabilistic IRN plausibilities. After step 2, this can be
   changed to use Plackett-Luce plausibilities instead.
4. Run `colab_partial_ao.ipynb` for an example of how to use the partial average
   overlap algorithm from [2].

## Citing this work

When using any part of this repository, make sure to cite both papers as follows:

```
@article{StutzTMLR2023,
    title={Conformal prediction under ambiguous ground truth},
    author={David Stutz and Abhijit Guha Roy and Tatiana Matejovicova and Patricia Strachan and Ali Taylan Cemgil and Arnaud Doucet},
    journal={Transactions on Machine Learning Research},
    issn={2835-8856},
    year={2023},
    url={https://openreview.net/forum?id=CAd6V2qXxc},
}
@article{DBLP:journals/corr/abs-2307-02191,
    author = {David Stutz and Ali Taylan Cemgil and Abhijit Guha Roy and Tatiana Matejovicova and Melih Barsbey and Patricia Strachan and Mike Schaekermann and Jan Freyberg and Rajeev Rikhye and Beverly Freeman and Javier Perez Matos and Umesh Telang and Dale R. Webster and Yuan Liu and Gregory S. Corrado and Yossi Matias and Pushmeet Kohli and Yun Liu and Arnaud Doucet and Alan Karthikesalingam},
    title = {Evaluating {AI} systems under uncertain ground truth: a case study
    in dermatology},
    journal = {CoRR},
    volume = {abs/2307.02191},
    year = {2023},
}
```

## License and disclaimer

Copyright 2023 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
