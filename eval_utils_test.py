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

"""Tests for evaluation utilities."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

import eval_utils


class EvalUtilsTest(parameterized.TestCase):

  def test_normalize_plausibilities(self):
    samples = jnp.array([
        [[1, 5, 3, 1]],
        [[0, 0, 0, 1]],
        [[0, 0, 0, 0]],
    ])
    expected_plausibilities = jnp.array([
        [[0.1, 0.5, 0.3, 0.1]],
        [[0, 0, 0, 1]],
        [[0, 0, 0, 0]],
    ])
    normalize_plausibilities = jax.jit(eval_utils.normalize_plausibilities)
    plausibilities = normalize_plausibilities(samples)
    np.testing.assert_array_almost_equal(
        plausibilities, expected_plausibilities
    )

  @parameterized.parameters([
      dict(
          expected_certainties=np.array([[0.2, 0.3, 0.4, 0.1], [0, 1, 0, 0]]),
          k=1,
      ),
      dict(
          expected_certainties=np.array([[0, 0.5, 0.3, 0.2], [0, 0, 0, 1]]),
          k=2,
      ),
  ])
  def test_rankk_certainties(self, expected_certainties, k):
    plausibilities = jnp.array([
        [
            # Majority is 2 across all plausibilities, but for one each, 0 or 1
            # are the top-1, so rank-1 certainty is [2/10, 3/10, 4/10, 1/10]
            # for all classes. Rank-2 certainty is [0, 5/10, 3/10, 2/10].
            [0, 0.1, 0.9, 0],
            [0, 0.1, 0.9, 0],
            [0, 0.1, 0.9, 0],
            [0, 0.1, 0.9, 0],
            [0, 0.9, 0.1, 0],
            [0, 0.9, 0, 0.1],
            [0, 0.9, 0.1, 0],
            [0.9, 0.1, 0, 0],
            [0.9, 0, 0, 0.1],
            [0, 0, 0.1, 0.9],
        ],
        # To have a second example, only rank-1 certainty is 1, 0 else.
        [[1, 4, 2, 3]] * 10,
    ])
    compute_certainties = jax.jit(
        functools.partial(eval_utils.rankk_certainties, k=k)
    )
    certainties = compute_certainties(plausibilities, jnp.arange(4))
    np.testing.assert_array_almost_equal(certainties, expected_certainties)

  def test_map_across_plausibilities(self):
    fn = lambda pred, plaus: jnp.sum(pred * plaus, axis=1)
    predictions = jnp.array([[0, 0, 1, 1], [1, 1, 0, 0]])
    plausibilities = jnp.array([
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        [
            [0.5, 0.5, 0, 0],
            [0, 0.5, 0.5, 0],
            [0, 0, 0.5, 0.5],
            [0.5, 0, 0, 0.5],
        ],
    ])
    expected_results = jnp.array([
        [0, 0, 1, 1],
        [1, 0.5, 0, 0.5],
    ])
    map_across_plausibilities = jax.jit(
        functools.partial(eval_utils.map_across_plausibilities, fn=fn)
    )
    results = map_across_plausibilities(predictions, plausibilities)
    np.testing.assert_array_almost_equal(results, expected_results)

  @parameterized.parameters([
      dict(num_readers=np.array([1, 2, 3, 4]), committee_size=1),
      dict(num_readers=np.array([4, 3, 2, 1]), committee_size=3),
      dict(num_readers=np.array([4, 3, 2, 1]), committee_size=10),
  ])
  def test_bootstrap_readers(self, num_readers, committee_size):
    bootstrap_readers = jax.jit(
        functools.partial(
            eval_utils.bootstrap_readers, committee_size=committee_size
        )
    )
    indices = bootstrap_readers(jax.random.PRNGKey(0), jnp.array(num_readers))
    num_cases = num_readers.shape[0]
    np.testing.assert_array_equal(indices.shape, (num_cases, committee_size))
    self.assertTrue(jnp.all(indices >= 0))
    for n in range(num_cases):
      self.assertTrue(jnp.all(indices[n] < num_readers[n]))

  def test_bootstrap_aggregated_rankings(self):
    rankings = jnp.array([
        [
            [3, 2, 1, 0],
            [0, 1, 2, 3],
        ],
        [
            [3, 2, 1, 0],
            [0, 1, 2, 3],
        ],
    ])
    groups = jnp.array([
        [
            [0, 1, -1, -1],
            [0, 0, 1, 1],
        ],
        [
            [0, 1, 2, 3],
            [-1, -1, -1, -1],
        ],
    ])
    num_examples, _, num_classes = rankings.shape
    num_readers = jnp.array([2, 1])
    committee_size = 3
    num_trials = 3
    bootstrap_aggregated_rankings = jax.jit(
        functools.partial(
            eval_utils.bootstrap_aggregated_rankings,
            committee_size=committee_size,
            num_trials=num_trials,
        )
    )
    samples = bootstrap_aggregated_rankings(
        jax.random.PRNGKey(0), rankings, groups, num_readers
    )
    np.testing.assert_array_equal(
        samples.shape, (num_examples, num_trials, num_classes)
    )

  def test_majority_vote(self):
    plausibilities = jnp.array([
        [
            [0, 2, 1],
            [0, 0.5, 1],
            [-1, 0, -0.5],
        ],
        [
            [0, 2, 3],
            [0, 0.5, 1],
            [-1, 0, -0.5],
        ],
    ])
    argmax_fn = functools.partial(jnp.argmax, axis=1)

    def rank2_fn(plausibilities):
      _, indices = jax.lax.top_k(plausibilities, k=2)
      return indices[:, -1]

    expected_majority_votes = [jnp.array([1, 2]), jnp.array([2, 1])]
    for fn, expected_majority_vote in zip(
        [argmax_fn, rank2_fn], expected_majority_votes
    ):
      majority_vote = jax.jit(
          functools.partial(eval_utils.majority_vote, fn=fn)
      )
      majority_votes = majority_vote(plausibilities)
      np.testing.assert_array_almost_equal(
          majority_votes, expected_majority_vote
      )


if __name__ == '__main__':
  absltest.main()
