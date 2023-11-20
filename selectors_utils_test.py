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

"""Tests for selectors_utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

import selectors_utils


class SelectorsUtilsTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(
          selectors=[[
              [[3], [1], [2]],
              [[0], [4, 1, 2]],
          ]],
          expected_error=selectors_utils.SelectorsStatus.NO_ERROR,
      ),
      dict(
          selectors=[[
              [[3.], [1], [2]],  # Invalid: float label.
              [[0], [4, 1, 2]],
          ]],
          expected_error=selectors_utils.SelectorsStatus.INVALID_TYPE,
      ),
      dict(
          selectors=[[
              [[3], [1], [2]],
              [[0], [4, 1, 2]],
              [],  # Invalid: empty reader.
          ]],
          expected_error=selectors_utils.SelectorsStatus.EMPTY_READER_OR_GROUP,
      ),
      dict(
          selectors=[[
              [[3], [1], [2]],
              [[0], [4, 1, 2]],
              [[]],  # Invalid: empty block/group.
          ]],
          expected_error=selectors_utils.SelectorsStatus.EMPTY_READER_OR_GROUP,
      ),
      dict(
          selectors=[[
              [[3], [1], [2]],
              [[0], [4, 1, 2]],
              [[0], [4, 1], [1, 3]],  # Invalid: duplicate label.
          ]],
          expected_error=selectors_utils.SelectorsStatus.DUPLICATES_IN_READER,
      ),
      dict(
          selectors=[[[269]], [[80], [135, 269]], [[80], [1, 269]], [[269]],
                     [[135, 269], [54, 151], [87]], [[269]], [[269]], [[80]],
                     [[135]]],
          expected_error=selectors_utils.SelectorsStatus.NO_ERROR,
      ),
      # Example with multiple instances: one valid, one with duplicate labels
      # as this is checked last.
      dict(
          selectors=[
              [[[3], [1], [2]]],
              [[[0], [4, 1], [1, 3]]],
          ],
          expected_error=selectors_utils.SelectorsStatus.DUPLICATES_IN_READER,
      ),
  ])
  def test_check_selectors(self, selectors, expected_error):
    self.assertEqual(
        selectors_utils.check_selectors(selectors, 288), expected_error)

  @parameterized.parameters([
      dict(
          selectors_expected=[[
              [[2, 3, 4, 5], [6, 7, 8]],
              [[0, 5], [1, 2, 3, 4]],
          ], [[[1, 2, 3, 4], [5, 6, 7, 8]]]],
          max_group_size=4),
      dict(
          selectors_expected=[[
              [[2, 3, 4], [5], [6, 7, 8]],
              [[0, 5], [1, 2, 3], [4]],
          ], [[[1, 2, 3], [4, 5, 6], [7, 8]]]],
          max_group_size=3),
      dict(
          selectors_expected=[[
              [[2], [3], [4], [5], [6], [7], [8]],
              [[0], [5], [1], [2], [3], [4]],
          ], [[[1], [2], [3], [4], [5], [6], [7], [8]]]],
          max_group_size=1)
  ])
  def test_limit_selectors_group_size(self, selectors_expected, max_group_size):
    selectors = [[
        [[2, 3, 4, 5], [6, 7, 8]],
        [[0, 5], [1, 2, 3, 4]],
    ], [[[1, 2, 3, 4, 5, 6, 7, 8]]]]
    selectors_returned = selectors_utils.limit_selectors_group_size(
        selectors, max_group_size)
    self._assert_selectors_equal(selectors_expected, selectors_returned)

  def test_pad_selectors(self):
    max_readers = 4
    reader1 = [[3], [1], [2]]
    reader2 = [[0], [4, 1, 2]]
    reader3 = [[1], [5, 1]]
    reader4 = [[2, 1, 5]]
    reader5 = [[1, 2, 3, 4, 5]]
    pad_reader = []
    selectors = [
        [reader1, reader2],
        [reader1],
        [reader1, reader2, reader3],
        [reader1, reader2, reader3, reader4],
        [reader1, reader2, reader3, reader4, reader5],
    ]
    selectors_expected = [
        [reader1, reader2, pad_reader, pad_reader],
        [reader1, pad_reader, pad_reader, pad_reader],
        [reader1, reader2, reader3, pad_reader],
        [reader1, reader2, reader3, reader4],
        [reader1, reader2, reader3, reader4],
    ]
    selectors_returned = selectors_utils.pad_selectors(selectors, max_readers)
    self._assert_selectors_equal(selectors_expected, selectors_returned)

  @parameterized.parameters([
      dict(
          rankings=[
              [
                  [0, 1, 2, 3],
                  [3, 0, 2, 1],
              ],
              [
                  [1, 3, 0, 2],
              ],
          ],
          groups=[
              [
                  [0, -1, -1, -1],
                  [0, 1, -1, -1],
              ],
              [
                  [0, 0, -1, -1],
              ],
          ],
      ),
      dict(
          rankings=[
              [
                  np.array([0, 1, 2, 3]),
                  np.array([3, 0, 2, 1]),
              ],
              [
                  np.array([1, 3, 0, 2]),
              ],
          ],
          groups=[
              [
                  np.array([0, -1, -1, -1]),
                  np.array([0, 1, -1, -1]),
              ],
              [
                  np.array([0, 0, -1, -1]),
              ],
          ],
      ),
  ])
  def test_pad_rankings(self, rankings, groups):
    num_examples = len(rankings)
    max_readers = 3
    num_classes = 4
    padded_rankings, padded_groups = selectors_utils.pad_rankings(
        rankings, groups, max_readers
    )
    np.testing.assert_array_equal(
        padded_rankings.shape, (num_examples, max_readers, num_classes)
    )
    np.testing.assert_array_equal(
        padded_groups.shape, (num_examples, max_readers, num_classes)
    )
    empty_ranking = jnp.arange(num_classes)
    empty_group = jnp.ones(4) * (-1)
    for n in range(num_examples):
      for r in range(max_readers):
        if r < len(rankings[n]):
          np.testing.assert_array_almost_equal(
              padded_rankings[n, r], jnp.array(rankings[n][r])
          )
          np.testing.assert_array_almost_equal(
              padded_groups[n, r], jnp.array(groups[n][r])
          )
        else:
          np.testing.assert_array_almost_equal(
              padded_rankings[n, r], empty_ranking
          )
          np.testing.assert_array_almost_equal(padded_groups[n, r], empty_group)

  def _assert_selectors_equal(self, s1, s2):
    self.assertEqual(len(s1), len(s2))
    for i in range(len(s1)):
      self.assertEqual(len(s1[i]), len(s2[i]))
      for j in range(len(s1[i])):
        self.assertEqual(len(s1[i][j]), len(s2[i][j]))
        for k in range(len(s1[i][j])):
          self.assertEqual(s1[i][j][k], s2[i][j][k])

  def test_repeat_selectors(self):
    reader1 = [[3], [1], [2]]
    reader2 = [[0], [4, 1, 2]]
    reader3 = [[1], [5, 1]]
    selectors = [
        [reader1, reader2],
        [reader1],
        [reader1, reader2, reader3],
    ]
    selectors_expected = [
        [reader1, reader1, reader1, reader2, reader2, reader2],
        [reader1, reader1, reader1],
        [
            reader1,
            reader1,
            reader1,
            reader2,
            reader2,
            reader2,
            reader3,
            reader3,
            reader3,
        ],
    ]
    selectors_returned = selectors_utils.repeat_selectors(selectors, 1)
    self._assert_selectors_equal(selectors, selectors_returned)
    selectors_returned = selectors_utils.repeat_selectors(selectors, 3)
    self._assert_selectors_equal(selectors_expected, selectors_returned)


if __name__ == '__main__':
  absltest.main()
