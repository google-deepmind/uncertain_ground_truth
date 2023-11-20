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

"""Tests for classification metrics."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np

import classification_metrics


class ClassificationMetricsTest(parameterized.TestCase):

  def test_aggregated_coverage(self):
    prediction_sets = np.array([[1, 1, 1, 0, 0], [0, 1, 0, 1, 0],
                                [0.5, -1, 1, 0, 0], [0, 0, 0, 1, 0]])
    distributions = np.repeat(np.array([[0.1, 0.2, 0.05, 0.4, 0.3]]), 4, axis=0)
    expected_result = np.array([0.35, 0.6, 0.15, 0.4])
    aggregated_coverage = classification_metrics.aggregated_coverage
    result = aggregated_coverage(prediction_sets, distributions)
    np.testing.assert_array_almost_equal(result, expected_result)

  def test_size(self):
    prediction_sets = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0],
                                [1, 1, 1, 1, 1], [0, 0, 0, 1, 1]])
    expected_result = np.array([0, 1, 5, 2])
    size = jax.jit(classification_metrics.size)
    result = size(prediction_sets)
    np.testing.assert_array_almost_equal(result, expected_result)

  @parameterized.parameters([
      dict(
          logits=np.array([[5, 3, 4, 4.5, 2], [-4, 1, 3, -1, -0.5]]),
          k=3,
          expected_prediction_sets=np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]]),
      ),
      # Examples with ties but different k:
      dict(
          logits=np.array([[5, 3, 4, 4, 2], [-4, -0.5, 3, -1, -0.5]]),
          k=3,
          expected_prediction_sets=np.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]]),
      ),
      dict(
          logits=np.array([[5, 3, 4, 4, 2], [-4, -0.5, 3, -1, -0.5]]),
          k=2,
          expected_prediction_sets=np.array([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0]]),
      ),
  ])
  def test_topk_sets(self, logits, k, expected_prediction_sets):
    topk_sets = jax.jit(
        functools.partial(classification_metrics.topk_sets, k=k)
    )
    prediction_sets = topk_sets(logits)
    np.testing.assert_array_almost_equal(
        prediction_sets, expected_prediction_sets
    )

  @parameterized.parameters([
      dict(
          logits=np.array([[5, 3, 4, 4.5, 2], [-4, 1, 3, -1, -0.5]]),
          k=3,
          expected_indicators=np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 1]]),
      ),
      # Examples with ties but different k:
      dict(
          logits=np.array([[5, 3, 4, 4, 2], [-4, -0.5, 3, -1, -0.5]]),
          k=3,
          expected_indicators=np.array([[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]),
      ),
      dict(
          logits=np.array([[5, 3, 4, 4, 2], [-4, -0.5, 3, -1, -0.5]]),
          k=2,
          expected_indicators=np.array([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0]]),
      ),
  ])
  def test_topk_indicators(self, logits, k, expected_indicators):
    topk_indicators = jax.jit(
        functools.partial(classification_metrics.topk_indicators, k=k)
    )
    indicators = topk_indicators(logits)
    np.testing.assert_array_almost_equal(indicators, expected_indicators)

  def test_aggregated_topk_accuracy(self):
    logits = np.array([[5, 4, 3, 2, 1], [0, 0.5, 0.1, 0.3, 0.2],
                       [1, -2, -0.1, -1, -3], [0.5, 0.9, -0.1, -1, 0]])
    distributions = np.repeat(np.array([[0.1, 0.2, 0.05, 0.4, 0.3]]), 4, axis=0)
    # Note the differen k used compared to above test.
    expected_result = np.array([0.75, 0.95, 0.75, 0.65])
    aggregated_topk_accuracy = jax.jit(
        functools.partial(classification_metrics.aggregated_topk_accuracy, k=4)
    )
    result = aggregated_topk_accuracy(logits, distributions)
    np.testing.assert_array_almost_equal(result, expected_result)


if __name__ == '__main__':
  absltest.main()
