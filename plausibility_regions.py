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

"""Conformal prediction on plausibilities produces plausibility regions."""

from typing import Tuple

import jax
import jax.numpy as jnp

import classification_metrics
import conformal_prediction


def expected_conformity_score(
    conformity_scores: jnp.ndarray, plausibilities: jnp.ndarray
) -> jnp.ndarray:
  """Computes an expected conformity score based on plausibilities.

  Args:
    conformity_scores: Conformity scores of shape `num_examples x num_classes`.
    plausibilities: Plausibilities of shape `num_examples x num_classes`.

  Returns:
    Expected conformity scores as `num_examples` shaped array.
  """
  return jnp.sum(conformity_scores * plausibilities, axis=1)


def calibrate_plausibility_regions(
    conformity_scores: jnp.ndarray, plausibilities: jnp.ndarray, alpha: float
) -> float:
  """Calibrate threshold for plausibility regions.

  Uses `conformal_prediction.conformal_quantile` to calibrate a threshold
  based on the `expected_conformity_score`. Threshold can be used to
  predict plausibility regions using `predict_plausibility_regions`.

  Args:
    conformity_scores: Conformity scores of shape `num_examples x num_classes`
      for validation examples.
    plausibilities: Plausibilities of shape `num_examples x num_classes` for
      validation examples.
    alpha: Confidence level in `[0,1]`.

  Returns:
    Calibrated threshold.
  """
  expected_conformity_scores = expected_conformity_score(
      conformity_scores, plausibilities
  )
  return conformal_prediction.conformal_quantile(
      expected_conformity_scores, alpha
  )


def check_plausibility_regions(
    conformity_scores: jnp.ndarray,
    plausibilities: jnp.ndarray,
    threshold: float,
) -> jnp.ndarray:
  """Check coverage for plausibilities given a threshold.

  Args:
    conformity_scores: Conformity scores of shape `num_examples x num_classes`
      for test examples.
    plausibilities: Plausibilities of shape `num_examples x num_classes` for
      test examples.
    threshold: Threshold from `calibrate_plausibility_regions`.

  Returns:
    Boolean array of shape `num_examples` indicating coverage of the
    provided plausibilities for each example.
  """
  conformity_scores = expected_conformity_score(
      conformity_scores, plausibilities
  )
  coverage = conformity_scores >= threshold
  return coverage


def predict_plausibility_regions(
    conformity_scores: jnp.ndarray,
    threshold: float,
    num_grid_points: int = 10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Predict plausibility regions using a grid.

  Args:
    conformity_scores: Test conformity scores of shape `num_examples x
      num_classes`.
    threshold: Threshold from `calibrate_plausibility_regions`.
    num_grid_points: Number of grid points to use.

  Returns:
    All plausibilities on the grid of shape `num_plausibilities x num_classes`
    and the corresponding coverages for the plausibility regions of each example
    as boolean array of shape `num_examples x num_plausibilities`.
  """

  def sample_plausibilities(num_classes):
    """Enumerate plausibilities on a `num_classes` dimensional grid."""
    linspace = jnp.linspace(0, 1, num_grid_points)
    grids = jnp.meshgrid(*[linspace] * num_classes)
    distributions = jnp.stack(tuple([g.reshape(-1) for g in grids]), axis=1)
    distributions = distributions[jnp.sum(distributions, axis=1) <= 1]
    return distributions

  _, num_classes = conformity_scores.shape
  plausibilities = sample_plausibilities(num_classes)

  def predict(scores):
    """Predict plausibility regions for example `n`."""
    predictions_n = jnp.repeat(
        jnp.expand_dims(scores, axis=0),
        plausibilities.shape[0],
        axis=0,
    )
    return check_plausibility_regions(predictions_n, plausibilities, threshold)

  return plausibilities, jax.vmap(predict, in_axes=(0))(conformity_scores)


def reduce_plausibilities_to_topk(
    plausibilities: jnp.ndarray, coverages: jnp.ndarray, k: int
) -> jnp.ndarray:
  """Reduce plausibility regions to top-k confidence sets.

  Given a plausibility region, represented by multiple plausibility samples
  per example, this function reduces this region to a single confidence set
  by considering the top-k classes of each plausibility sample.

  Args:
    plausibilities: Plausibilities as `num_examples x num_samples x num_classes`
      array.
    coverages: The coverages obtained from `predict_plausibility_regions` as
      `num_examples x num_plausibilities` shaped boolean array.
    k: The number of top classes to consider for all plausibility samples.

  Returns:
    A `num_examples x num_classes` shaped array for confidence sets.
  """
  num_examples, _ = coverages.shape
  topk_sets = classification_metrics.topk_sets(plausibilities, k=k)

  def reduce_to_topk(n):
    """Helper to reduce all plausibility top-k sets to single confidence set."""
    return jnp.clip(
        jnp.sum(topk_sets * jnp.expand_dims(coverages[n], axis=1), axis=0), 0, 1
    )

  return jax.vmap(reduce_to_topk)(jnp.arange(num_examples))
