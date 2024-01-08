# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Classification metrics for evaluation, focusing on accuracy and coverage.

We provide accuracy and coverage implementations for classification,
assuming inputs $x$ with corresponding labels $y$ and a (true) posterior
distribution $p(y|x)$. In most classification settings, we assume $p(y|x)$
to be crisp, i.e., nearly one-hot, in the sense that there is a clear label
$k$ that obtains the highest probability $p(y = k|x) \approx 1$. With
ambiguous ground truth, however, we want to work with general posterior
distributions $p(y|x)$ in which case it is unclear how to compute
metrics such as accuracy and coverage.

Additionally, we consider prediction sets $C(x)$ instead of single predictions.
If $C(x)$ is a top-$k$ prediction, we follow common convention all call
metrics top-$k$ accuracy. If $C(x)$ can have variable size depending on the
example, we follow conformal prediction convention and call the metrics
coverage, i.e., coverage = "variable-k" accuracy.

In order to compute accuracy and coverage for arbitrary $p(y|x)$, we define
**aggregated** coverage as follows:

\Pr(y \in C(x))
= \sum_y \int \delta[y \in C(x)] p(y, x) dx
= \int \left(\sum_y \delta[y \in C(x)] p(y, x)\right) dx
= \int \left(\sum_y \delta[y \in C(x)] p(y|x)p(x)\right) dx
= \int \mathbb{E}_{p(y|x)}[\delta[y \in C(x)] p(x) dx

If $p(y|x)$ is one-hot, aggregated coverage coincides with the standard coverage
definition and is a binary event per example, i.e., the single true label is
either covered in $C(x)$ or not. For arbitrary $p(y|x)$, aggregated coverage is
a
continuous event per example quantifying the probability mass of $p(y|x)$
covered by the prediction set $C(x)$. aggregated top-k accuracy is defined as
aggregated coverage for fixed top-$k$ prediction sets $C(x)$. In practice, this
is
implemented by using samples for the integral over
$p(x)$ and explicitly computing the expectation
$\mathbb{E}_{p(y|x)}[\delta[y \in C(x)]$, as shown above.

Ideal coverage is implemented by expecting the `prediction_sets` $C(x)$ as 0-1
encoded array of shape `num_examples x num_classes` and the posterior
distribution (`distributions`) $p(y|x)$ as `num_examples x num_classes` shaped
array where each row sums to $1$. The former is enforced, i.e., 0-1 encoded
prediction sets, while the posterior distribution is not enforced to be a
probability distribution. This is to allow implementation of alternative
semantics such as handling ties (example below). Then, the expectation
$\mathbb{E}_{p(y|x)}[\delta[y \in C(x)]$ can be computed by multiplying
`prediction_sets` and `distributions` element-wise and summing over the
`num_classes` axes.
"""

import jax
import jax.numpy as jnp


def valid_prediction_sets(prediction_sets: jnp.ndarray) -> jnp.ndarray:
  """Ensures that confidence sets are 0-1 encoded.

  Args:
    prediction_sets: confidence sets as num_examples x num_classes array

  Returns:
    Confidence sets as 0-1 array
  """
  return (prediction_sets > 0).astype(int)


def aggregated_coverage(
    prediction_sets: jnp.ndarray, distributions: jnp.ndarray, clip: bool = True
) -> jnp.ndarray:
  r"""Compute aggregated coverage.

  This computes the expectation $\mathbb{E}_{p(y|x)}[\delta[y \in C(x)]$
  for each example, as described above.

  Args:
    prediction_sets: Prediction sets as 0-1 encoded `num_examples x num_classes`
      array; the array is binarized and converted to int to obtain valid
      prediction sets.
    distributions: The ground truth posterior distributions over classes of
      shape `num_examples x num_classes`.
    clip: Whether to clip coverage per example to [0, 1].

  Returns:
    aggregated coverage value of shape `num_examples` with values in [0, 1] if
    `clip` is True.
  """
  prediction_sets = valid_prediction_sets(prediction_sets)
  coverages = jnp.sum(prediction_sets * distributions, axis=1)
  if clip:
    return jnp.clip(coverages, 0, 1)
  return coverages


def size(prediction_sets: jnp.ndarray) -> jnp.ndarray:
  """Compute prediction set size.

  Args:
    prediction_sets: confidence sets on test set as 0-1 array

  Returns:
    Confidence set sizes of shape `num_examples` with values in
    `{0, ..., num_classes}`.
  """
  return jnp.sum(prediction_sets, axis=1)


def _make_prediction_sets(
    logits: jnp.ndarray, index: jnp.ndarray
) -> jnp.ndarray:
  """Helper to easily create multiple-hot encoded confidence sets."""
  prediction_set = jnp.zeros(logits.shape, int)
  return prediction_set.at[index].set(1)


def topk_sets(
    logits: jnp.ndarray,
    k: int,
) -> jnp.ndarray:
  """Converts logits to a top-k prediction set.

  Note that `jax.lax.top_k`, which is used for this, resolves ties
  by order. That is, if for one row in `logits` there are multiple
  values with the same value, `jax.lax.top_k` will select the first (smallest)
  indices.

  Args:
    logits: Predicted logits or probabilities of shape `num_examples x
      num_classes`.
    k: Number of top elements to consider per example.

  Returns:
    Predictions sets corresponding to top-k selection in 0-1 encoding
    of shape `num_examples x num_classes.
  """
  _, indices = jax.lax.top_k(logits, k)
  make_prediction_sets = jax.vmap(_make_prediction_sets, in_axes=(0, 0))
  return make_prediction_sets(logits, indices)


def topk_indicators(
    logits: jnp.ndarray,
    k: int,
) -> jnp.ndarray:
  """Converts logits to an indicator set of the rank-k classes.

  Args:
    logits: Predicted logits or probabilities of shape `num_examples x
      num_classes`.
    k: Rank of classes to retrieve as indicators.

  Returns:
    Indicator sets corresponding to rank-k selection in 0-1 encoding
    of shape `num_examples x num_classes.
  """
  _, indices = jax.lax.top_k(logits, k)
  indices = indices[:, -1]
  make_prediction_sets = jax.vmap(_make_prediction_sets, in_axes=(0, 0))
  return make_prediction_sets(logits, indices)


def aggregated_topk_accuracy(
    logits: jnp.ndarray, distributions: jnp.ndarray, k: int, clip: bool = True
) -> jnp.ndarray:
  """Compute aggregated top-k accuracy.

  Args:
    logits: Predicted logits or probabilities of shape `num_examples x
      num_classes`.
    distributions: The ground truth posterior distributions over classes of
      shape `num_examples x num_classes`.
    k: Number of top elements to consider per example.
    clip: Whether to clip coverage per example to [0, 1].

  Returns:
    Aggregated top-k accuracy of shape `num_examples` with values in [0, 1] if
    `clip` is True.
  """
  prediction_sets = topk_sets(logits, k)
  return aggregated_coverage(prediction_sets, distributions, clip)
