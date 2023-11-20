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

"""Conformal predictor tests."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import scipy.special

import conformal_prediction


class ConformalPredictionTest(parameterized.TestCase):

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
      dict(num_examples=10000, num_classes=10),
  ])
  def test_calibrate_predict_threshold(self, num_examples, num_classes):
    val_labels, val_probabilities = self._get_examples(
        num_examples, num_classes, 2
    )
    test_labels, test_probabilities = self._get_examples(
        num_examples, num_classes, 2
    )

    alpha = 0.05
    threshold = conformal_prediction.calibrate_threshold(
        val_probabilities, val_labels, alpha)
    test_confidence_sets = conformal_prediction.predict_threshold(
        test_probabilities, threshold)
    coverage = test_confidence_sets[jnp.arange(test_confidence_sets.shape[0]),
                                    test_labels]
    self.assertAlmostEqual(jnp.mean(coverage), 1 - alpha, places=1)

  @parameterized.parameters([
      dict(
          val_conformity_scores=np.array([
              [0.1, 0.1, 0.1],
              [0.1, 0.5, 0.4],
              [0.2, 0.3, 0.7],
          ]),
          val_labels=np.array([0, 1, 2]),  # 0.1, 0.5 and 0.7.
          test_conformity_scores=np.array([
              [0.1, 0.2, 0.3, 0.4],
              [0.5, 0.6, 0.7, 0.8],
              [0.9, 1, 0, -0.1],
          ]),
          expected_ranks=np.array([
              [1, 1, 1, 1],
              [2, 2, 3, 3],
              [3, 3, 0, 0],
          ]),
      ),
  ])
  def test_compute_ranks(
      self,
      val_conformity_scores,
      val_labels,
      test_conformity_scores,
      expected_ranks,
  ):
    ranks = conformal_prediction.compute_ranks(
        val_conformity_scores, val_labels, test_conformity_scores
    )
    np.testing.assert_array_almost_equal(ranks, expected_ranks)

  @parameterized.parameters([
      dict(num_examples=1000, num_classes=10),
      dict(num_examples=1000, num_classes=100),
  ])
  def test_compute_p_values(self, num_examples, num_classes):
    # Probabilities have to be one-hot for the p-value tests.
    val_labels, val_probabilities = self._get_examples(
        num_examples, num_classes, 100
    )
    test_labels, test_probabilities = self._get_examples(
        num_examples, num_classes, 100
    )

    p_values = conformal_prediction.compute_p_values(
        val_probabilities, val_labels, test_probabilities
    )
    np.testing.assert_array_equal(p_values.shape, (num_examples, num_classes))
    correct_p_values = p_values[jnp.arange(num_examples), test_labels]
    np.testing.assert_almost_equal(
        correct_p_values, np.ones_like(correct_p_values)
    )
    correct_mask = jax.nn.one_hot(test_labels, num_classes)
    incorrect_p_values = p_values * (1 - correct_mask)
    incorrect_p_values += correct_mask / (num_examples + 1)
    np.testing.assert_almost_equal(
        incorrect_p_values,
        np.ones_like(incorrect_p_values) / (num_examples + 1),
    )

  @parameterized.parameters([
      dict(num_examples=10000, num_classes=10),
  ])
  def test_compute_p_values_predictions(self, num_examples, num_classes):
    # Logits are random in [0, 1], so adding 2 to the true class
    # will ensure that the argmax is always the true label.
    val_labels, val_probabilities = self._get_examples(
        num_examples, num_classes, 2
    )
    test_labels, test_probabilities = self._get_examples(
        num_examples, num_classes, 2
    )

    p_values = conformal_prediction.compute_p_values(val_probabilities,
                                                     val_labels,
                                                     test_probabilities)
    predicted_test_labels = jnp.argmax(p_values, axis=1)
    np.testing.assert_equal(
        np.array(test_labels), np.array(predicted_test_labels))


if __name__ == "__main__":
  absltest.main()
