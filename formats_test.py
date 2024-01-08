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

"""Tests for formats."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

import formats


class FormatsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(
          selectors=[[
              [[3], [1], [2]],
              [[0], [4, 1, 2]],
          ]],
          expected_rankings=np.array([[
              [3, 1, 2, 0, 4],
              [0, 4, 1, 2, 3],
          ]]),
          expected_groups=np.array([[
              [0, 1, 2, -1, -1],
              [0, 1, 1, 1, -1],
          ]]),
          num_classes=5,
      ),
  ])
  def test_convert_selectors_to_rankings(self, selectors, expected_rankings,
                                         expected_groups, num_classes):
    rankings, groups = formats.convert_selectors_to_rankings(
        selectors, num_classes)
    np.testing.assert_array_almost_equal(rankings, expected_rankings)
    np.testing.assert_array_almost_equal(groups, expected_groups)

  def test_convert_prediction_sets_to_rankings(self):
    conformity_scores = jnp.array([[0, 1, 2, 3], [3, 1, 0, 2]])
    prediction_sets = jnp.array([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ])
    expected_rankings = jnp.array([
        [3, 2, 1, 0],
        [0, 3, 1, 2],
    ])
    expected_groups = jnp.array([
        [0, -1, -1, -1],
        [0, 1, -1, -1],
    ])
    rankings, groups = formats.convert_prediction_sets_to_rankings(
        conformity_scores, prediction_sets
    )
    np.testing.assert_array_almost_equal(rankings, expected_rankings)
    np.testing.assert_array_almost_equal(groups, expected_groups)

  def test_convert_rankings_to_prediction_sets(self):
    rankings = jnp.array([
        [3, 2, 1, 0],
        [0, 3, 1, 2],
    ])
    groups = jnp.array([
        [0, -1, -1, -1],
        [0, 1, -1, -1],
    ])
    expected_prediction_sets = jnp.array([
        [0, 0, 0, 1],
        [1, 0, 0, 1],
    ])
    prediction_sets = formats.convert_rankings_to_prediction_sets(
        rankings, groups
    )
    np.testing.assert_array_almost_equal(
        prediction_sets, expected_prediction_sets
    )


if __name__ == '__main__':
  absltest.main()
