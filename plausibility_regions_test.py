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

"""Tests for plausibility regions."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

import plausibility_regions


class PlausibilityRegionsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(
          plausibilities=np.array([[0, 0, 0.5, 0.5], [1, 0, 0, 0]]),
          reference_conformity_scores=np.array([0.75, 0.4]),
      ),
      dict(
          plausibilities=np.array([[0.1, 0.9, 0, 0], [0.2, 0.2, 0.3, 0.2]]),
          reference_conformity_scores=np.array([3.6, 0.22]),
      ),
  ])
  def test_expected_conformity_scores(
      self, plausibilities, reference_conformity_scores
  ):
    conformity_scores = jnp.array([[0, 4, 1, 0.5], [0.4, 0.1, 0.2, 0.3]])
    expected_conformity_scores = plausibility_regions.expected_conformity_score(
        conformity_scores, jnp.array(plausibilities)
    )
    np.testing.assert_array_almost_equal(
        expected_conformity_scores, reference_conformity_scores
    )

  @parameterized.parameters([
      dict(
          threshold=0.5,
          expected_coverages=np.array([
              [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
              [0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
          ]),
      ),
      dict(
          threshold=1,
          expected_coverages=np.array([
              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
          ]),
      ),
      dict(
          threshold=0.0,
          expected_coverages=np.array([[1] * 10] * 3),
      ),
  ])
  def test_predict_plausibility_regions(self, threshold, expected_coverages):
    grid_points = 3
    conformity_scores = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    expected_plausibilities = jnp.array([
        [0, 0, 0],
        [0, 0, 0.5],
        [0, 0, 1],
        [0.5, 0, 0],
        [0.5, 0, 0.5],
        [1, 0, 0],
        [0, 0.5, 0],
        [0, 0.5, 0.5],
        [0.5, 0.5, 0],
        [0, 1, 0],
    ])
    plausibilities, coverages = (
        plausibility_regions.predict_plausibility_regions(
            conformity_scores, threshold, grid_points
        )
    )
    np.testing.assert_array_almost_equal(
        plausibilities, expected_plausibilities
    )
    np.testing.assert_array_almost_equal(
        coverages, expected_coverages.astype(bool)
    )

  @parameterized.parameters([
      dict(
          k=1,
          expected_confidence_sets=np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1]]),
      ),
      dict(
          k=2,
          expected_confidence_sets=np.array([[1, 1, 1], [0, 1, 1], [1, 1, 1]]),
      ),
  ])
  def test_reduce_plausibilities_topk(self, k, expected_confidence_sets):
    coverages = jnp.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]])
    plausibilities = jnp.array([
        [1, 0, 0],
        [0, 0.2, 0.8],
        [0.2, 0.7, 0.1],
    ])
    confidence_sets = plausibility_regions.reduce_plausibilities_to_topk(
        plausibilities, coverages, k
    )
    np.testing.assert_array_almost_equal(
        confidence_sets, expected_confidence_sets
    )


if __name__ == '__main__':
  absltest.main()
