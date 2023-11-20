# Copyright 2023 DeepMind Technologies Limited
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

"""Utilities for evaluation and certainty computation with plausibilities.

In our setting, (normalized) plausibilities represent categorical distributions
over classes. Sampling multiple plausibilities per case allows us to quantify
the annotation uncertainty, i.e., the impact of disagreement among
annotations on, e.g., the ranking or top-k classes. We generally assume
the plausibilities to not exhibit ties; if this is the case, we assume
that ties are resolved randomly before using the utilities in this file,
especially when computing certainties.
"""

import functools
from typing import Callable

import jax
import jax.numpy as jnp

import irn as aggregation


def normalize_plausibilities(samples: jnp.ndarray) -> jnp.ndarray:
  """Normalizes samples to plausibilities along last dimension.

  Args:
    samples: Samples, i.e., unnormalized plausibilities, e.g., as `num_examples
      x num_samples x num_classes` array to be normalized.

  Returns:
    Plausibilities, e.g., as `num_examples x num_samples x num_classes` array,
    which are the input samples normalized along the last dimension.
  """
  return samples / (jnp.expand_dims(jnp.sum(samples, axis=-1), axis=-1) + 1e-8)


def rankk_certainties(
    plausibilities: jnp.ndarray, risk_groups: jnp.ndarray, k: int = 1
) -> jnp.ndarray:
  """Computes rank-k certainties.

  Certainties are computed by counting frequencies of how often
  the rank-k (e.g., argmax for rank-1) label of the plausibilities is j,
  for all classes j.

  Args:
    plausibilities: `num_examples x num_samples x num_classes` array of
      plausibilities; rank-k certainties can only be computed in a meaningful
      way using `num_samples > 1`.
    risk_groups: Mapping from all `num_classes` conditions to the conditions or
      groups, e.g., risk levels, of interest; by default this could be
      `jnp.arange(num_classes)` but it could also be a mapping form conditions
      to risk levels.
    k: Rank to compute certainties for.

  Returns:
    Rank-k certainties as `num_examples x num_classes` array.
  """
  _, _, num_classes = plausibilities.shape
  # top_k is slightly faster than sorting, but will be significantly slower
  # than using the argmax for k=1.
  if k > 1:
    _, rankk_labels = jax.lax.top_k(plausibilities, k)
    rankk_labels = rankk_labels[:, :, -1]
  else:
    rankk_labels = jnp.argmax(plausibilities, axis=2)

  def _accumulate_certainties(k):
    return jnp.mean(risk_groups[rankk_labels] == k, axis=1)

  accumulate_certainties = jax.vmap(_accumulate_certainties)
  return accumulate_certainties(jnp.arange(num_classes)).T


def map_across_plausibilities(
    predictions: jnp.ndarray,
    plausibilities: jnp.ndarray,
    fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
  """Apply a specific metric across all plausibilities per example.

  Args:
    predictions: Predictions of shape `num_examples x num_classes`.
    plausibilities: `num_samples` Gibbs or IRN plausibilities for each case,
      resulting in an array of shape `num_examples x num_samples x num_classes`.
    fn: A callable, e.g., a  metric, taking predictions of shape `num_examples x
      num_classes` and plausibilities of shape `num_examples x num_classes` as
      input, i.e., one plausibility vector per example.

  Returns:
    Function `fn` mapped across plausibilities, returning the output of `fn`
    accumulated for `num_samples` plausibilities. If `fn` returns one value
    per example, i.e., an array of shape `num_examples`, the overall output
    will be `num_samples x num_examples`.
  """
  return jax.vmap(fn, (None, 1), 1)(predictions, plausibilities)


def bootstrap_readers(
    rng: jnp.ndarray, num_readers: jnp.ndarray, committee_size: int
) -> jnp.ndarray:
  """Bootstrap committees readers for each example.

  Args:
    rng: Random key.
    num_readers: Number of readers for each example in terms of a `num_examples`
      shaped array.
    committee_size: Number of readers in each bootstrapped committee.

  Returns:
    Array with reader indices of shape
    `num_examples x committee_size`.
  """
  num_examples = num_readers.shape[0]
  return jax.random.randint(
      rng,
      shape=(num_examples, committee_size),
      minval=jnp.zeros((num_examples, 1)),
      maxval=jnp.expand_dims(num_readers, axis=1),
  )


def bootstrap_aggregated_rankings(
    rng: jnp.ndarray,
    rankings: jnp.ndarray,
    groups: jnp.ndarray,
    num_readers: jnp.ndarray,
    committee_size: int,
    num_trials: int,
) -> jnp.ndarray:
  """Computes aggregated ranking samples from bootstrapped readers.

  Uses `bootstrap_readers` to sample `max_readers` for each case. This is
  repeated `num_trials` times and for each trial the sampled readers'
  rankings are aggregated. This results in `num_trials` numbers of
  aggregated plausibilities for each case, similar to Gibbs samples.

  Args:
    rng: Random key.
    rankings: `num_examples x num_classes` shaped array of full rankings.
    groups: `num_examples x num_classes` shaped array of groups that indicate
      the structure of partial rankings together with `rankings`; also see
      `aggregation.irn.aggregate_irn`.
    num_readers: Number of readers for each example in terms of a `num_examples`
      shaped array.
    committee_size: Number of readers in each bootstrapped committee.
    num_trials: Number of trials to sample readers for.

  Returns:
    Array of aggregated plausibilities of shape
    `num_examples x num_trials x num_classes`.
  """
  irn_samples = []
  num_examples = rankings.shape[0]
  rngs = jax.random.split(rng, num_trials)
  for t in range(num_trials):
    indices = bootstrap_readers(rngs[t], num_readers, committee_size)
    # We want to select multiple readers for each case as indicated by indices
    # which is of shape `num_examples x committee_size`. In order to do this
    # via advanced indexing, we also need to index the first axis (cases)
    # using indices with two axes: jnp.arange(num_examples)[:, None] is
    # automatically broadcast for this.
    ranking_samples = rankings[jnp.arange(num_examples)[:, None], indices, :]
    group_samples = groups[jnp.arange(num_examples)[:, None], indices, :]
    irn_samples.append(
        aggregation.aggregate_irn(ranking_samples, group_samples)
    )
  return jnp.array(irn_samples).transpose((1, 0, 2))


def majority_vote(
    plausibilities: jnp.ndarray, fn: Callable[[jnp.ndarray], jnp.ndarray]
) -> jnp.ndarray:
  """Majority votes the result of a function over plausibilities.

  For fn = jnp.argmax, for example, this function computes the majority-voted
  top-1 label across all plausibilities per example.

  Note that ties in the votes will be resolved as in jnp.argmax, i.e., the
  top-1 label with lowest index in the example above.

  Args:
    plausibilities: `num_samples` Gibbs or IRN plausibilities for each case,
      resulting in an array of shape `num_examples x num_samples x num_classes`.
    fn: A callable, e.g., jnp.argmax, taking a single plausibility vector of
      shape `num_classes` and returning a scalar.

  Returns:
    Majority voted results of fn across plausibilities for each example,
    of shape `num_examples`.
  """

  def _majority_vote(plausibilities):
    _, num_classes = plausibilities.shape
    return jnp.argmax(jnp.bincount(fn(plausibilities), length=num_classes))

  return jax.vmap(_majority_vote)(plausibilities)


majority_vote_argmax = functools.partial(
    majority_vote, fn=functools.partial(jnp.argmax, axis=1)
)
