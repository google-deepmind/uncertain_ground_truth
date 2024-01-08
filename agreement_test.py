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

"""Tests for agreement computation."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

import agreement


class AgreementTest(parameterized.TestCase):

  def test_leave_one_reader_out_agreement(self):
    rankings = jnp.array([[[0, 1, 2]] * 3] * 2)
    groups = jnp.array([
        [
            [0, -1, -1],
            [0, 1, -1],
            [0, 0, -1],
        ],
        [
            [0, -1, -1],
            [0, 1, -1],
            [-1, -1, -1],
        ],
    ])
    num_readers = jnp.array([3, 2])

    expected_group_overlap = jnp.array([
        [1, 1.5, 1.5],
        [0.5, 0.5, 0],  # 0.5 because the last empty reader is still included.
    ])

    # Basically looks at how many non-negative entries in groups overlap
    # between readers and averages that across readers.
    def average_group_overlap(
        left_out_rankings,
        left_out_groups,  # pylint: disable=unused-argument
        other_rankings,
        other_groups,  # pylint: disable=unused-argument
        num_readers,
    ):  # pylint: disable=unused-argument
      left_out_groups = jnp.repeat(left_out_groups, 2, axis=1)
      group_overlap = jnp.logical_and(left_out_groups >= 0, other_groups >= 0)
      return jnp.mean(jnp.sum(group_overlap, axis=2), axis=1)

    group_overlap = agreement.leave_one_reader_out_agreement(
        rankings, groups, num_readers, average_group_overlap
    )
    np.testing.assert_array_almost_equal(group_overlap, expected_group_overlap)

  def test_leave_one_reader_out_coverage_agreement(self):
    rankings = jnp.array([[[0, 1, 2]] * 3] * 2)
    groups = jnp.array([[[0, 1, -1]] * 3] * 2)
    num_readers = jnp.array([3, 2])
    standard_accuracy_agreement = (
        agreement.leave_one_reader_out_coverage_agreement(
            rankings, groups, num_readers
        )
    )
    np.testing.assert_array_equal(standard_accuracy_agreement.shape, (2, 3))
    self.assertTrue(jnp.all(standard_accuracy_agreement <= 1))
    self.assertTrue(jnp.all(standard_accuracy_agreement >= 0))


if __name__ == '__main__':
  absltest.main()
