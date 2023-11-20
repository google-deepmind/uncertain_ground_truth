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

"""Gaussian mixture toy dataset with ambiguous ground truth.

Essentially, the model allows sampling examples with a single true ground truth
label, evaluate these examples under the model's distribution to obtain
a distribution over ground truth labels and sample rankings that simulate
human expert annotations.
"""

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp


class PRNGSequence:
  """Iterator of JAX random keys similar to Haiku."""

  def __init__(self, key_or_seed: Union[jnp.ndarray, int]):
    """Creates a new :class:`PRNGSequence`."""
    if isinstance(key_or_seed, int):
      key_or_seed = jax.random.PRNGKey(key_or_seed)
    # A seed value may also be passed as an int32-typed scalar ndarray.
    elif isinstance(key_or_seed, jnp.ndarray):
      key_or_seed = jax.random.PRNGKey(key_or_seed)
    else:
      raise ValueError('Expected integer seed or jax.random.PRNGKey.')
    self._key = key_or_seed

  def __next__(self) -> jnp.ndarray:
    self._key, rng = jax.random.split(self._key)
    return rng

  next = __next__


class GaussianToyDataset:
  """Toy dataset with overlapping Gaussians to simulate ambiguous ground truth.

  Essentially, this is a Gaussian mixture model defined by
  class weights and dimensionality. The individual Gaussian means are sampled
  from a standard Gaussian with specified sigma such that sigma controls
  the likelihood of overlap.

  We assume a diagonal covariate matrix such that the multi-dimensional
  Gaussians are separable by dimensions.
  """

  def __init__(
      self,
      key_sequence: PRNGSequence,
      class_weights: jnp.ndarray,
      class_sigmas: Union[float, jnp.ndarray],
      dimensionality: int,
      sigma: float = 1,
  ):
    """Constructor.

    Args:
      key_sequence: sequence of random keys such as hk.PRNGSequence
      class_weights: weights of individual classes, does not have to be
        normalized
      class_sigmas: standard deviation of individual Gaussian, the same for all
        dimensions
      dimensionality: dimensionality of data
      sigma: standard deviation to sample the mean of individual Gaussians from
    """

    if class_weights.ndim != 1:
      raise ValueError('Expecting a one-dimensional vector for class weights.')
    if class_weights.size <= 0:
      raise ValueError('Cannot use zero classes.')
    if isinstance(class_sigmas, float) or isinstance(class_sigmas, int):
      class_sigmas = jnp.ones(class_weights.shape) * float(class_sigmas)
    if class_sigmas.ndim != 1:
      raise ValueError(
          'Experiment a one-dimensional vector for class standard deviations.')
    if class_sigmas.size != class_weights.size:
      raise ValueError('Expecting class_sigmas of same shape as class_weights.')
    if sigma <= 0:
      raise ValueError('Expecting a positive sigma.')
    if dimensionality <= 0:
      raise ValueError('Expecting a positive dimensionality.')

    self.key_sequence = key_sequence
    """ (hk.PRNGSequence) Key sequence for random generators. """

    self.num_classes = class_weights.shape[0]
    """ (int) Number of classes. """

    self.class_probabilities = class_weights / jnp.sum(class_weights)
    """ (jnp.array) Class probabilities. """

    self.dimensionality = dimensionality
    """ (int) Gaussian dimensions. """

    self.sigma = sigma
    """ (float) Standard deviation of Gaussian means. """

    self.means = None
    """ (jnp.array) Gaussian means per class. """

    self.sigmas = jnp.tile(
        jnp.expand_dims(class_sigmas, axis=1), (1, self.dimensionality))
    """ (jnp.array) Gaussian standard deviations per class. """

    self.build()

  def build(self):
    """Build the individual classes' Gaussians."""
    means = []
    for _ in range(self.num_classes):
      mean = jax.random.normal(
          next(self.key_sequence), shape=(self.dimensionality,))
      mean = mean * self.sigma + 0.5
      means.append(mean)
    self.means = jnp.array(means)

  def sample_points(self, num_examples: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample from the model.

    Samples ground truth classes first, then samples
    points from the corresponding Gaussian.

    Args:
      num_examples: number of examples to sample

    Returns:
      Sampled points and corresponding ground truths
    """
    if num_examples <= 0:
      raise ValueError('Can only sample a positive number of examples.')
    labels = jax.random.categorical(
        next(self.key_sequence),
        jnp.log(self.class_probabilities),
        shape=(num_examples,))

    def sample_single(d, n, keys):
      k = labels[n]
      mean = self.means[k, d]
      sigma = self.sigmas[k, d]
      value = jax.random.normal(keys[d * num_examples + n])
      return value * sigma + mean

    sample_d = jax.vmap(sample_single, in_axes=(0, None, None))
    sample_dk = jax.vmap(sample_d, in_axes=(None, 0, None))
    keys = jax.random.split(
        next(self.key_sequence), num_examples * self.dimensionality)
    points = sample_dk(
        jnp.arange(self.dimensionality), jnp.arange(num_examples), keys)
    return points, labels

  def evaluate_points(self, points: jnp.ndarray) -> jnp.ndarray:
    """Evaluate probability of points.

    Evaluated the probability density function (pdf) corresponding to each class
    for all points. Overall pdf is then obtained by summing over classes
    weighted by class weights and potentially normalizing.

    Args:
      points: examples to evaluate

    Returns:
      Probabilities per class
    """
    if len(points.shape) != 2:
      raise ValueError('Expecting points of shape num_points x dim.')
    if points.shape[1] != self.dimensionality:
      raise ValueError('Expected points with dimensionality %d.' %
                       self.dimensionality)
    # We aim for a N x d x K matrix where entry n, d, k holds
    # the probability of d-th dimension in example n evaluated
    # under class k.
    # Overall probability is then obtained by taking the product over d
    # and the sum over k.
    def evaluate_single(n, d, k):
      mean = self.means[k, d]
      sigma = self.sigmas[k, d]
      return jax.scipy.stats.norm.logpdf((points[n, d] - mean) / sigma)

    evaluate_k = jax.vmap(evaluate_single, in_axes=(None, None, 0))
    evaluate_dk = jax.vmap(evaluate_k, in_axes=(None, 0, None))
    evaluate_ndk = jax.vmap(evaluate_dk, in_axes=(0, None, None))
    logits = evaluate_ndk(
        jnp.arange(points.shape[0]), jnp.arange(self.dimensionality),
        jnp.arange(self.num_classes))
    # Sum because we deal with log-probabilities.
    logits = jnp.sum(logits, axis=1)
    # This takes care of stability, because logits can get very large
    # in high dimensions in which case the exponential would be infinity.
    logits -= jnp.expand_dims(jnp.max(logits, axis=1), axis=1)
    probabilities = jnp.exp(logits)
    # Note that these are per-class probabilities without taking the prior
    # into account.
    probabilities *= jnp.expand_dims(self.class_probabilities, axis=0)
    probabilities /= jnp.expand_dims(jnp.sum(probabilities, axis=1), axis=1)
    return probabilities

  def _sample_simple_rankings(self, probabilities,
                              readers: jnp.ndarray) -> jnp.ndarray:
    """Helper to sample simple, full length and non-partial rankings."""

    # We need a vmap here since jax.random.choice does not allow
    # sampling multiple values from different categorical distributions.
    num_examples = probabilities.shape[0]
    num_rankings = readers.shape[0]

    def sample_individual(n, m, keys):
      return jax.random.choice(
          keys[m * num_examples + n],
          self.num_classes,
          replace=False,
          p=probabilities[n, m],
          shape=(self.num_classes,))

    sample_m = jax.vmap(sample_individual, in_axes=(None, 0, None))
    sample_nm = jax.vmap(sample_m, in_axes=(0, None, None))
    keys = jax.random.split(
        next(self.key_sequence), 2 * num_examples * num_rankings)
    return sample_nm(jnp.arange(num_examples), jnp.arange(num_rankings), keys)

  def _sample_partial_rankings(
      self,
      probabilities: jnp.ndarray,
      rankings: jnp.ndarray,
      expected_length: Optional[int] = None,
      grouping_threshold: Optional[int] = None,
  ) -> jnp.ndarray:
    """Helper to sample partial rankings from _sample_simple_rankings."""

    num_examples, num_rankings, num_classes = probabilities.shape
    indices = jnp.tile(
        jnp.expand_dims(
            jnp.expand_dims(jnp.arange(self.num_classes), axis=0), axis=0),
        (num_examples, num_rankings, 1))

    # Merge close classes based on probabilities.
    if grouping_threshold is not None:
      flattened_rankings = rankings.reshape(-1, num_classes)
      flattened_probabilities = probabilities.reshape(-1, num_classes)
      num_cases = num_examples * num_rankings
      default_indices = jnp.indices((num_cases, num_classes))
      sorted_probabilities = flattened_probabilities[default_indices[0],
                                                     flattened_rankings]
      next_probabilities = jnp.roll(sorted_probabilities, shift=1, axis=1)
      sorted_differences = sorted_probabilities - next_probabilities
      groups = (jnp.abs(sorted_differences) >
                grouping_threshold + 1e-6).astype(int)
      groups = groups.at[jnp.arange(num_cases), 0].set(0)
      groups = jnp.cumsum(groups, axis=1)
      groups = groups.reshape(num_examples, num_rankings, num_classes)
    else:
      groups = indices

    # Limit the length of the rankings.
    if expected_length is not None:
      lengths = jnp.maximum(
          1,
          jax.random.poisson(
              next(self.key_sequence), expected_length,
              (num_examples * num_rankings,)))
      lengths = jnp.reshape(lengths, (num_examples, num_rankings, 1))

      cut_off = (indices >= lengths)
      groups = groups.at[cut_off].set(-1)
    return groups

  def sample_rankings(
      self,
      probabilities: jnp.ndarray,
      reader_sharpness: jnp.ndarray,
      expected_length: Optional[int] = None,
      grouping_threshold: Optional[float] = None
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample human rankings based on ground truth probabilities.

    Args:
      probabilities: probabilities of samples to sample rankings for
      reader_sharpness: quality of readers in the form of the sharpness of a
        Dirichlet distribution that is used to sample the actual observed
        probabilities from, high values indicate good reader quality
      expected_length: expected length of rankings
      grouping_threshold: if class probabilities of two consecutive classes are
        closer than grouping_threshold they will be put in the same block in a
        partial ordering.

    Returns:
      num_rankings partial rankings for each example in points
    """

    if probabilities.ndim != 2:
      raise ValueError('Expecting probabilities with two dimensions.')
    if probabilities.shape[1] != self.num_classes:
      raise ValueError('Expecting probabilities for %d classes.' %
                       self.num_classes)
    if reader_sharpness.ndim != 1:
      raise ValueError('Expecting a one-dimensional error of reader sharpness.')
    if reader_sharpness.size <= 0:
      raise ValueError('Expected at least one reader.')
    if expected_length is not None and expected_length <= 0:
      raise ValueError('Expected length has to be positive.')
    if grouping_threshold is not None and grouping_threshold <= 0:
      raise ValueError('Grouping threshold has to be positive.')

    num_readers = reader_sharpness.shape[0]
    num_examples = probabilities.shape[0]
    probabilities = jnp.tile(
        jnp.expand_dims(probabilities, axis=1), (1, num_readers, 1))

    def sample_observed_probabilities(r):
      """Helper to sample from Dirichlet distribution per reader."""
      sharpness = reader_sharpness[r]
      return jax.random.dirichlet(
          next(self.key_sequence),
          alpha=probabilities[:, r, :] * sharpness,
          shape=(num_examples,))

    sample_observed_probabilities_r = jax.vmap(
        sample_observed_probabilities, in_axes=(0,))
    observed_probabilities = sample_observed_probabilities_r(
        jnp.arange(num_readers)).transpose((1, 0, 2))

    rankings = self._sample_simple_rankings(observed_probabilities,
                                            reader_sharpness)
    groups = self._sample_partial_rankings(observed_probabilities, rankings,
                                           expected_length, grouping_threshold)
    return rankings, groups

  def vary_number_of_readers(
      self,
      groups: jnp.ndarray,
      expected_readers: int,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Enforces variable number of readers.

    Essentially sets the `groups` array to `-1` for a variable number of readers
    for each example. Makes sure that at least one reader per example remains.

    Args:
      groups: Groups array from `sample_rankings` indicating partial ranking
        groups.
      expected_readers: Expected number of readers.

    Returns:
      Updated groups array indicating blocks in partial rankings with
      some rows set to -1 indicating missing readers.
    """
    num_examples, num_readers, _ = groups.shape
    var_num_readers = jax.random.poisson(
        next(self.key_sequence), expected_readers, (num_examples,))
    var_num_readers = jnp.minimum(num_readers, jnp.maximum(1, var_num_readers))

    def discard_reader(n):
      """Helper to discard remaining readers on a per example basis."""
      group = groups[n]
      mask = jnp.expand_dims(
          jnp.arange(num_readers) < var_num_readers[n], axis=1).astype(int)
      group = mask * group + (1 - mask) * jnp.ones(group.shape) * (-1)
      return group

    discard_reader_n = jax.vmap(discard_reader, in_axes=(0))
    return discard_reader_n(jnp.arange(num_examples)), var_num_readers
