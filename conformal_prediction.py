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

"""Basic conformal predictors.

Implementations of two approaches to conformal prediction independent
of the conformity scores used:
1. By calibrating a threshold using `calibrate_threshold` and
2. by computing p-values using `compute_p_values`.
"""

import jax.numpy as jnp


def conformal_quantile(scores: jnp.ndarray, alpha: float) -> float:
  """Computed a corrected quantile for conformal inference.

  Given the confidence level alpha and `num_examples` scores, the corrected
  quantile is the `(1 + 1 / num_examples) * alpha` quantile.

  Args:
    scores: Conformity scores of shape `num_examples`.
    alpha: Confidence level in `[0,1]`.

  Returns:
    Corrected conformal quantile.
  """
  return float(
      jnp.quantile(
          scores,
          jnp.floor(alpha * (scores.shape[0] + 1)) / scores.shape[0],
          axis=None,
          method='midpoint',
      )
  )


def calibrate_threshold(conformity_scores: jnp.ndarray, labels: jnp.ndarray,
                        alpha: float) -> float:
  """Corrected quantile for conformal prediction.

  Args:
    conformity_scores: Conformity scores for all classes of shape `num_examples
      x num_classes`.
    labels: Ground truth labels of shape `num_examples`.
    alpha: Confidence level in `[0,1]`.

  Returns:
    Threshold for conformal prediction using `predict_threshold`.
  """
  true_conformity_scores = conformity_scores[
      jnp.arange(conformity_scores.shape[0]), labels]
  return conformal_quantile(true_conformity_scores, alpha)


def predict_threshold(conformity_scores: jnp.ndarray,
                      threshold: float) -> jnp.ndarray:
  """Predict confidence sets based on threshold.

  Args:
    conformity_scores: Conformity scores for test examples of shape
      `num_examples x num_classes`.
    threshold: Calibrated threshold from `calibrate_threshold`.

  Returns:
    Confidence sets of shape `num_examples x num_classes` indicating membership
    by 1, 0 otherwise.
  """
  return (conformity_scores >= threshold).astype(int)


def compute_ranks(
    val_conformity_scores: jnp.ndarray,
    val_labels: jnp.ndarray,
    test_conformity_scores: jnp.ndarray,
) -> jnp.ndarray:
  """Computes ranks of test conformity scores in validation scores.

  This is used for p-values computation by dividing the rank, which is
  equivalent to the number of validation conformity scores that are smaller
  than the test score, by the number of validation examples plus one.

  Can also be used for effective number of tests computation when
  combining multiple dependent p-values.

  Args:
    val_conformity_scores: Conformity scores of validation examples of shape
      `num_val_examples x num_classes`.
    val_labels: Validation labels of shape `num_val_examples`.
    test_conformity_scores: Conformity score sof test examples of shape
      `num_test_examples x num_classes`.

  Returns:
    Ranks for test examples of shape `num_test_examples x num_classes`.
  """
  # We need to test all test examples with all validation examples,
  # so we add a singleton dimension after the example dimension.
  test_conformity_scores = jnp.expand_dims(test_conformity_scores, axis=1)
  val_conformity_scores = val_conformity_scores[
      jnp.arange(val_conformity_scores.shape[0]), val_labels]
  # Shape validation conformity scores the same way as the test one
  # but relying on broadcasting to only use the conformity scores
  # corresponding to the true labels
  val_conformity_scores = val_conformity_scores.reshape((1, -1, 1))
  ranks = test_conformity_scores >= val_conformity_scores
  ranks = jnp.sum(ranks, axis=1)
  return ranks


def compute_p_values(
    val_conformity_scores: jnp.ndarray,
    val_labels: jnp.ndarray,
    test_conformity_scores: jnp.ndarray,
) -> jnp.ndarray:
  """Computes p-values based on validation and test conformity scores.

  Computes p-values by dividing the ranks from `compute_ranks` by
  `num_val_examples + 1`.

  Args:
    val_conformity_scores: Conformity scores of validation examples of shape
      `num_val_examples x num_classes`.
    val_labels: Validation labels of shape `num_val_examples`.
    test_conformity_scores: Conformity score sof test examples of shape
      `num_test_examples x num_classes`.

  Returns:
    P-values for test examples of shape `num_test_examples x num_classes`.
  """
  val_examples = val_conformity_scores.shape[0]
  nominator = 1 + compute_ranks(
      val_conformity_scores, val_labels, test_conformity_scores
  )
  denominator = 1 + val_examples
  return nominator / denominator


def predict_p_values(p_values: jnp.ndarray, alpha: float) -> jnp.ndarray:
  """Predict confidence sets using p-values.

  Args:
    p_values: p-values for test examples from `compute_p_values` of shape
      `num_examples`.
    alpha: Confidence level in `[0,1]`.

  Returns:
    Confidence sets of shape `num_examples x num_classes` with 1 indicating
    membership, 0 otherwise.
  """
  return (p_values >= alpha).astype(int)
