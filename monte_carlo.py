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

"""Monte Carlo conformal predictors.

Mirroring the approaches in `conformal_prediction`, there are two approaches
of performing Monte Carlo conformal prediction:
* By calibrating a threshold using `calibrate_mc_threshold` and then
  using `conformal_prediction.predict_threshold`;
* or by computing p-values using `compute_mc_p_values`.

The core idea of Monte Carlo conformal prediction is to sample from the
provided smooth labels (i.e., distributions over classes given as smooth
ground truth) `num_samples` times and then perform conformal prediction
on `num_examples * num_samples` validation examples where the
conformity scores are repeated `num_samples` times. As this breaks
exchangeability, the p-values need to be corrected by using
`compute_mc_ecdf_p_values` if a finite-sample coverage guarantee of `1 - alpha`
is desired. Otherwise, coverage `1 - 2*alpha` is guaranteed.
"""

import jax
import jax.numpy as jnp

import conformal_prediction
import p_value_combination


def sample_mc_labels(
    rng: jnp.ndarray,
    conformity_scores: jnp.ndarray,
    smooth_labels: jnp.ndarray,
    num_samples: int = 10,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Monte Carlo approach to sample labels from plausibilities.

  Uses the given smooth labels to sample multiple hard labels for each
  example. The obtained conformity scores and labels can then be used
  for conformal prediction using a threshold or p-values. Each example
  is repeated exactly `num_samples` times.

  This approach does not provide a coverage guarantee for `num_samples > 1`
  as the test conformity scores and the sampled calibration conformity scores
  are not exchangeable (due to the repetitions). For a version with
  coverage guarantee see `compute_random_mc_scores_and_labels` below.

  Args:
    rng: Random key.
    conformity_scores: Conformity scores of validation examples of shape
      `num_examples x num_classes`.
    smooth_labels: Smooth label distribution for validation examples of shape
      `num_examples x num_classes`.
    num_samples: Number of samples of ground truth labels per example.

  Returns:
    Sampled conformity scores and hard labels of shapes
    `num_samples x num_examples x num_classes` and
    `num_samples x num_examples`.
  """
  num_examples = conformity_scores.shape[0]
  labels = jax.random.categorical(
      rng,
      logits=jnp.log(smooth_labels + 1e-8),
      shape=(num_samples, num_examples),
  )
  conformity_scores = jnp.repeat(
      jnp.expand_dims(conformity_scores, axis=0), num_samples, axis=0
  )
  return conformity_scores, labels


def mc_conformal_quantile(
    scores: jnp.ndarray, num_examples: int, num_samples: int, alpha: float
) -> float:
  """Computed a corrected quantile for Monte Carlo conformal inference.

  Args:
    scores: Conformity scores of shape `num_examples * num_samples`.
    num_examples: Number of i.i.d. examples to consider.
    num_samples: Number of Monte Carlo samples included in `scores`.
    alpha: Confidence level in `[0,1]`.

  Returns:
    Corrected conformal quantile.
  """
  quantile = (
      jnp.floor(alpha * num_samples * (num_examples + 1)) - num_samples + 1
  )
  quantile /= num_examples * num_samples
  return float(jnp.quantile(scores, quantile, method='midpoint'))


def calibrate_mc_threshold(
    rng: jnp.ndarray,
    conformity_scores: jnp.ndarray,
    smooth_labels: jnp.ndarray,
    alpha: float,
    num_samples: int = 10,
) -> float:
  """Calibrates a threshold using Monte Carlo conformal prediction.

  Args:
    rng: Random key for `sample_mc_labels`.
   conformity_scores: Conformity scores of validation examples of shape
     `num_examples x num_classes`.
    smooth_labels: Smooth label distribution for validation examples of shape
      `num_examples x num_classes`.
    alpha: Confidence level in `[0,1]`.
    num_samples: Number of samples of ground truth labels per example.

  Returns:
    Conformal quantile of conformity scores corresponding to Monte Carlo
    sampled labels from `smooth_labels`.
  """
  num_examples, num_classes = conformity_scores.shape
  mc_conformity_scores, mc_labels = sample_mc_labels(
      rng, conformity_scores, smooth_labels, num_samples=num_samples
  )
  mc_conformity_scores = mc_conformity_scores.reshape(-1, num_classes)
  mc_labels = mc_labels.reshape(-1)
  true_mc_conformity_scores = mc_conformity_scores[
      jnp.arange(mc_conformity_scores.shape[0]), mc_labels
  ]
  return mc_conformal_quantile(
      true_mc_conformity_scores, num_examples, num_samples, alpha
  )


def compute_mc_p_values(
    rng: jnp.ndarray,
    val_conformity_scores: jnp.ndarray,
    val_smooth_labels: jnp.ndarray,
    test_conformity_scores: jnp.ndarray,
    num_samples: int = 10,
) -> jnp.ndarray:
  """Compute p-values using Monte Carlo sampled labels.

  Args:
    rng: Random key for `sample_mc_labels`.
    val_conformity_scores: Conformity scores of validation examples of shape
      `num_val_examples x num_classes`.
    val_smooth_labels: Smooth label distribution for validation examples of
      shape `num_val_examples x num_classes`.
    test_conformity_scores: Conformity scores of test examples of shape
      `num_test_examples x num_classes`.
    num_samples: Number of samples of ground truth labels per example.

  Returns:
    P-values for test examples of shape
    `num_samples x num_test_examples x num_classes`.
  """
  mc_conformity_scores, mc_labels = sample_mc_labels(
      rng, val_conformity_scores, val_smooth_labels, num_samples=num_samples
  )

  def compute_p_values(conformity_scores: jnp.ndarray, labels: jnp.ndarray):
    """Compute p-values for the `m` Monte Carlo sample."""
    return conformal_prediction.compute_p_values(
        conformity_scores, labels, test_conformity_scores
    )

  return jax.vmap(compute_p_values, in_axes=(0, 0))(
      mc_conformity_scores, mc_labels
  )


def compute_mc_ecdf_p_values(
    rng: jnp.ndarray,
    val_conformity_scores: jnp.ndarray,
    val_smooth_labels: jnp.ndarray,
    test_conformity_scores: jnp.ndarray,
    num_samples: int = 10,
    split: float = 0.5,
) -> jnp.ndarray:
  """Compute p-values using `compute_mc_p_values` and ECDF-correct them.

  Args:
    rng: Random key for `sample_mc_labels` and splitting the validation scores
      and labels according to `split`.
    val_conformity_scores: Conformity scores of validation examples of shape
      `num_val_examples x num_classes`.
    val_smooth_labels: Smooth label distribution for validation examples of
      shape `num_val_examples x num_classes`.
    test_conformity_scores: Conformity score sof test examples of shape
      `num_test_examples x num_classes`.
    num_samples: Number of samples of ground truth labels per example.
    split: Fraction of validation examples to use for calibration such that `1 -
      split` fraction of the validation examples will be used for the ECDF
      correction of p-values.

  Returns:
    Corrected p-values for test examples of shape
    `num_test_examples x num_classes`.
  """
  split_rng, mc_rng, est_rng = jax.random.split(rng, 3)
  # Split validation examples into actual calibration and ECDF estimation
  # examples.
  val_examples, num_classes = val_conformity_scores.shape
  test_examples, _ = test_conformity_scores.shape
  permutation = jax.random.permutation(split_rng, val_examples)
  cal_examples = int(val_examples * split)
  est_examples = val_examples - cal_examples
  cal_conformity_scores = val_conformity_scores[permutation[:cal_examples]]
  cal_smooth_labels = val_smooth_labels[permutation[:cal_examples]]
  est_conformity_scores = val_conformity_scores[permutation[cal_examples:]]
  est_smooth_labels = val_smooth_labels[permutation[cal_examples:]]

  est_p_values = compute_mc_p_values(
      mc_rng,
      cal_conformity_scores,
      cal_smooth_labels,
      est_conformity_scores,
      num_samples,
  )
  # We average them and sample new labels from est_smooth_labels.
  est_p_values = jnp.mean(est_p_values, axis=0)
  est_labels = jax.random.categorical(
      est_rng,
      logits=jnp.log(est_smooth_labels + 1e-8),
      shape=(est_smooth_labels.shape[0],),
  )
  est_p_values = est_p_values[jnp.arange(est_examples), est_labels]

  # We can use the same key.
  test_p_values = compute_mc_p_values(
      mc_rng,
      cal_conformity_scores,
      cal_smooth_labels,
      test_conformity_scores,
      num_samples,
  )
  test_p_values = jnp.mean(test_p_values, axis=0)

  corrected_test_p_values = p_value_combination.combine_ecdf_p_values(
      est_p_values, test_p_values.reshape(-1)
  ).reshape(test_examples, num_classes)
  return corrected_test_p_values
