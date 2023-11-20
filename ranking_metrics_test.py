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

"""Tests for the ranking metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

import ranking_metrics


class RankingMetricsTest(parameterized.TestCase):

  def _get_average_overlap_test_data(
      self) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    predicted_rankings = jnp.array([
        [2, 0, 1, 4, 3],
        [2, 0, 1, 4, 3],
        [2, 0, 1, 4, 3],
        [3, 4, 1, 0, 2],
        [3, 4, 1, 0, 2],
        [3, 4, 1, 0, 2],
    ])
    rankings = jnp.array([
        [0, 1, 2, 4, 3],  # (0/1 + 1/2 + 3/3) / 3 = 0.5.
        [1, 0, 2, 4, 3],  # (0/1 + 1/2 + 3/3 + 4/4) / 4 = 0.625.
        [1, 0, 2, 4, 3],  # (0/1 + 1/2 + 3/3 + 4/4 + 5/5) / 5 = 0.7.
        [4, 1, 3, 0, 2],  # (0/1)/ = 0.0.
        [3, 4, 1, 0, 2],  # (1/1 + 2/2)/ = 1.0.
        [3, 4, 1, 0, 2],  # max_length = 0 --> return 0.0.
    ])
    max_lengths = jnp.array([3, 4, 5, 1, 2, 0])
    expected_ao = jnp.array([0.5, 0.625, 0.7, 0.0, 1.0, 0.0])

    return predicted_rankings, rankings, max_lengths, expected_ao

  def test_average_overlap(self):
    (predicted_rankings, rankings, max_lengths,
     expected_ao) = self._get_average_overlap_test_data()

    average_overlap = jax.jit(ranking_metrics.average_overlap)
    np.testing.assert_array_almost_equal(
        average_overlap(
            predicted_rankings,
            rankings,
            max_lengths,
        ),
        expected_ao,
    )

  @parameterized.parameters([
      dict(
          normalize=False,
          expected_ao=np.array(
              [0.9375, 0.5, 0.5, 0.6666667, 0.4444445, 0.625, 0.0, 0.0, 0.0]),
      ),
      dict(
          normalize=True,
          expected_ao=np.array([
              1.0, 0.5773503, 0.5222330, 0.8164966, 0.4776652, 0.625, 0.0, 0.0,
              0.0
          ]),
      ),
  ])
  def test_partial_average_overlap(self, normalize, expected_ao):

    predicted_rankings = jnp.array([
        [0, 1, 2, 3],  # AO([0123, 0213], [0123, 0213]) = 0.9375
        [0, 1, 2, 3],  # AO([01], [20, 02]) = (0.75 + 0.25)/2.
        [0, 1, 2, 3],  # AO([012], [201, 210]) = (0.5 + 0.5)/2.
        [0, 1, 2, 3],  # AO([012], [012, 021, 102, 120, 201, 210]) =
        # (1.0 + 0.83 + 0.66 + 0.5 + 0.5 + 0.5)/6.
        [2, 0, 1, 3],  # AO([201, 210], [012, 013]) =
        # (0.5 + 0.5, 0.39 + 0.39)/4.
        [2, 0, 1, 3],  # AO([0123], [2013]) = 0.625 --> equivalent to AO.
        [2, 0, 1, 3],  # max_length = 0 --> return 0.0.
        [2, 0, 1, 3],  # predicted_groups all -1 --> return 0.0.
        [2, 0, 1, 3],  # groups all -1 --> return 0.0.
    ])

    predicted_groups = jnp.array([
        [0, 1, 1, -1],
        [0, 1, 2, -1],
        [0, 1, 2, -1],
        [0, 1, 2, -1],
        [0, 1, 1, 2],
        [0, 1, 2, 3],
        [0, 1, 1, 2],
        [-1, -1, -1, -1],
        [0, 1, 1, 2],
    ])

    rankings = jnp.array([
        [0, 1, 2, 3],
        [2, 0, 1, 3],
        [2, 0, 1, 3],
        [2, 0, 1, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ])

    groups = jnp.array([
        [0, 1, 1, -1],
        [0, 0, 1, 2],
        [0, 1, 1, 2],
        [0, 0, 0, 1],
        [0, 1, -1, -1],
        [0, 1, 2, 3],
        [0, 1, -1, -1],
        [0, 1, -1, -1],
        [-1, -1, -1, -1],
    ])

    max_lengths = jnp.array([4, 2, 3, 3, 3, 4, 0, 3, 3])

    partial_average_overlap = jax.jit(
        ranking_metrics.partial_average_overlap, static_argnames='normalize'
    )
    np.testing.assert_array_almost_equal(
        partial_average_overlap(
            predicted_rankings,
            predicted_groups,
            rankings,
            groups,
            max_lengths,
            normalize=normalize,
        ),
        expected_ao,
    )

  def test_build_partial_ranking_matrix(self):
    ranking = jnp.array([4, 2, 0, 1, 3, 5, 7, 6])
    groups = jnp.array([0, 0, 1, 2, 2, 2, -1, -1])

    expected_partial_ranking_matrix = jnp.array([
        [0.0, 0.0, 1 / 2, 0.0, 1 / 2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1 / 2, 0.0, 1 / 2, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1 / 3, 0.0, 1 / 3, 0.0, 1 / 3, 0.0, 0.0],
        [0.0, 1 / 3, 0.0, 1 / 3, 0.0, 1 / 3, 0.0, 0.0],
        [0.0, 1 / 3, 0.0, 1 / 3, 0.0, 1 / 3, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1 / 2, 1 / 2],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1 / 2, 1 / 2],
    ])

    build_partial_ranking_matrix = jax.jit(
        ranking_metrics._build_partial_ranking_matrix
    )

    np.testing.assert_array_almost_equal(
        build_partial_ranking_matrix(ranking, groups),
        expected_partial_ranking_matrix,
    )


if __name__ == '__main__':
  absltest.main()
