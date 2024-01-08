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

"""Conformal predictor tests."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

import conformal_prediction
import monte_carlo


class MonteCarloTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    np.random.seed(0)

  def _get_examples(
      self, num_examples: int, num_classes: int, dominance: float
  ):
    labels = jnp.array(np.random.randint(0, num_classes, (num_examples)))
    probabilities = np.random.random((labels.shape[0], np.max(labels) + 1))
    probabilities[np.arange(probabilities.shape[0]), labels] += dominance
    probabilities = scipy.special.softmax(probabilities, axis=1)
    return jnp.array(labels), jnp.array(probabilities)

  @parameterized.parameters([
      dict(num_examples=1000, num_classes=10, num_samples=1),
      dict(num_examples=1000, num_classes=10, num_samples=10),
  ])
  def test_sample_mc_labels_calibrate_mc_threshold(
      self, num_examples, num_classes, num_samples
  ):
    val_labels, val_probabilities = self._get_examples(
        num_examples, num_classes, 10
    )
    test_labels, test_probabilities = self._get_examples(
        num_examples, num_classes, 10
    )

    alpha = 0.05
    rng = jax.random.PRNGKey(0)
    mc_val_probabilities, mc_val_labels = monte_carlo.sample_mc_labels(
        rng, val_probabilities, val_probabilities, num_samples
    )
    np.testing.assert_array_equal(
        mc_val_probabilities.shape, (num_samples, num_examples, num_classes)
    )
    np.testing.assert_array_equal(
        mc_val_labels.shape, (num_samples, num_examples)
    )
    sampled_val_probabilities = mc_val_probabilities.reshape(-1, num_classes)
    sampled_val_labels = mc_val_labels.reshape(-1)

    if num_samples == 1:
      np.testing.assert_array_equal(sampled_val_labels, val_labels)

    expected_threshold = conformal_prediction.calibrate_threshold(
        sampled_val_probabilities, sampled_val_labels, alpha
    )

    threshold = monte_carlo.calibrate_mc_threshold(
        rng, val_probabilities, val_probabilities, alpha, num_samples
    )
    self.assertAlmostEqual(threshold, expected_threshold, places=4)

    test_confidence_sets = conformal_prediction.predict_threshold(
        test_probabilities, threshold)
    coverage = test_confidence_sets[jnp.arange(test_confidence_sets.shape[0]),
                                    test_labels]
    self.assertAlmostEqual(jnp.mean(coverage), 1 - alpha, places=1)

  @parameterized.parameters([
      dict(method=monte_carlo.compute_mc_p_values, correction=1.0),
      dict(method=monte_carlo.compute_mc_ecdf_p_values, correction=0.0),
  ])
  def test_compute_mc_p_values(self, method, correction):
    # Tests whether Monte Carlo CP is equivalent to standard CP
    # in the non-ambiguous case.
    num_examples = 1000
    num_classes = 10
    num_samples = 10
    _, val_probabilities = self._get_examples(num_examples, num_classes, 100)
    test_labels, test_probabilities = self._get_examples(
        num_examples, num_classes, 100
    )
    rng = jax.random.PRNGKey(0)
    mc_p_values = method(
        rng,
        val_probabilities,
        val_probabilities,
        test_probabilities,
        num_samples,
    )
    if mc_p_values.ndim < 3:
      mc_p_values = mc_p_values.reshape(1, -1, num_classes)
    for p_values in mc_p_values:
      np.testing.assert_array_equal(p_values.shape, (num_examples, num_classes))
      correct_p_values = p_values[jnp.arange(num_examples), test_labels]
      np.testing.assert_almost_equal(
          correct_p_values, np.ones_like(correct_p_values)
      )
      correct_mask = jax.nn.one_hot(test_labels, num_classes)
      incorrect_p_values = p_values * (1 - correct_mask)
      # Without ECDF correction, incorrect p-values are 1 / (num_examples + 1);
      # With ECDF correctin they tend to just be zero.
      incorrect_p_values += correction * correct_mask / (num_examples + 1)
      np.testing.assert_almost_equal(
          incorrect_p_values,
          correction * np.ones_like(incorrect_p_values) / (num_examples + 1),
      )


if __name__ == "__main__":
  absltest.main()
