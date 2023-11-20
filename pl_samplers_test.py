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

"""Tests for plackett_luce."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from pl_samplers import GibbsSamplerPlackettLuce


class SamplersTest(parameterized.TestCase):
  gibbs_samplers = {
      jit_strategy: GibbsSamplerPlackettLuce(jit_strategy=jit_strategy)
      for jit_strategy in ["jit_per_reader", "jit_per_iteration", "no_jit"]
  }

  def _get_test_gibbs_data(self):
    lam = jnp.array([1.0, 0.2, 2.0, 0.1, 4.0, 0.3, 1.0, 2.0])
    rankings_1 = jnp.array([2, 1, 0, 3, 4, 7, 6, 5])
    rankings_2 = jnp.array([7, 1, 4, 3, 2, 0, 5, 6])
    rankings = jnp.vstack((rankings_1, rankings_2))
    selectors_1 = [[2, 1], [0], [3, 4, 7]]
    selectors_2 = [[7], [1], [4, 3], [2]]
    selectors = [selectors_1, selectors_2]
    return rankings, selectors, lam

  def test_sample_tau_given_lam_and_rankings(self):
    gibbs_sampler = self.gibbs_samplers["jit_per_reader"]
    lam = jnp.array([1.0, 0.2, 2.0, 0.1])
    rankings = jnp.array([2, 1, 0, 3])
    key = jax.random.PRNGKey(0)
    tau = gibbs_sampler._sample_tau_given_lam_and_rankings(key, lam, rankings)
    np.testing.assert_array_almost_equal(rankings, jnp.argsort(tau))

  def test_sample_perm_given_lam_and_partial_rankings(self):
    gibbs_sampler = self.gibbs_samplers["jit_per_reader"]
    rankings, selectors, lam = self._get_test_gibbs_data()
    key = jax.random.PRNGKey(0)
    rankings_returned = (
        gibbs_sampler._sample_perm_given_lam_and_partial_rankings(
            key, rankings, selectors, lam
        )
    )
    r1_returned, r2_returned = rankings_returned[0, :], rankings_returned[1, :]
    self.assertCountEqual(jnp.array([1, 2]), r1_returned[:2])
    self.assertEqual(0, r1_returned[2])
    self.assertCountEqual(jnp.array([3, 4, 7]), r1_returned[3:6])
    self.assertCountEqual(jnp.array([5, 6]), r1_returned[6:])

    self.assertEqual(7, r2_returned[0])
    self.assertEqual(1, r2_returned[1])
    self.assertCountEqual(jnp.array([3, 4]), r2_returned[2:4])
    self.assertEqual(2, r2_returned[4])
    self.assertCountEqual(jnp.array([0, 5, 6]), r2_returned[5:])

  def test_sample_from_block_posterior(self):
    gibbs_sampler = self.gibbs_samplers["jit_per_reader"]
    phi = jnp.array([-1, 4.0, 3.0, 2.0])
    logsumexp_phi = jax.nn.logsumexp(phi)
    key = jax.random.PRNGKey(0)
    selector = [0, 1, 3]
    selector_returned = gibbs_sampler._sample_from_block_posterior(
        key, phi, logsumexp_phi, selector
    )
    selector_expected = jnp.array([1, 3, 0])
    np.testing.assert_array_almost_equal(selector_returned, selector_expected)

  def test_get_denoms(self):
    gibbs_sampler = self.gibbs_samplers["jit_per_reader"]
    lam = jnp.array([3.0, 2.0, 4.0])
    phi = jnp.log(lam)
    denoms_expected = jnp.log(jnp.array([3 + 2 + 4, 2 + 4, 4]))

    var_get_denoms = jax.jit(gibbs_sampler._get_denoms)
    denoms_returned = var_get_denoms(phi)
    np.testing.assert_array_almost_equal(
        denoms_expected, denoms_returned, decimal=5
    )

  def test_initialize_rankings(self):
    gibbs_sampler = self.gibbs_samplers["jit_per_reader"]
    _, selectors, lam = self._get_test_gibbs_data()
    num_classes = len(lam)
    key = jax.random.PRNGKey(0)
    rankings_returned = gibbs_sampler._initialize_rankings(
        key, selectors, num_classes
    )
    r1_returned, r2_returned = rankings_returned[0, :], rankings_returned[1, :]
    self.assertCountEqual(jnp.array([1, 2]), r1_returned[:2])
    self.assertEqual(0, r1_returned[2])
    self.assertCountEqual(jnp.array([3, 4, 7]), r1_returned[3:6])
    self.assertCountEqual(jnp.array([5, 6]), r1_returned[6:])

    self.assertEqual(7, r2_returned[0])
    self.assertEqual(1, r2_returned[1])
    self.assertCountEqual(jnp.array([3, 4]), r2_returned[2:4])
    self.assertEqual(2, r2_returned[4])
    self.assertCountEqual(jnp.array([0, 5, 6]), r2_returned[5:])

  def test_sample_lam_given_tau(self):
    gibbs_sampler = self.gibbs_samplers["jit_per_reader"]
    tau = jnp.array([[10.0, 0.1, 100.0, 15.0], [12.0, 0.2, 120.0, 16.0]])
    shape_lam, rate_lam = jnp.ones((4,)), jnp.ones((4,))
    lam = gibbs_sampler._sample_lam_given_tau(
        jax.random.PRNGKey(0), tau, shape_lam, rate_lam
    )
    self.assertGreater(lam[1], lam[2])

  def test_gibbs_sampler_pl_iteration(self):
    gibbs_sampler = self.gibbs_samplers["jit_per_reader"]
    rankings, selectors, lam = self._get_test_gibbs_data()
    key = jax.random.PRNGKey(0)
    shape_lam, rate_lam = jnp.ones((8,)), jnp.ones((8,))
    _, lam_returned, rank_returned = gibbs_sampler._gibbs_sampler_pl_iteration(
        key, lam, rankings, selectors, shape_lam, rate_lam
    )
    for l in lam_returned:
      self.assertGreater(l, 0)
    self.assertGreater(lam_returned[1], lam_returned[2])

    for r in rank_returned.flatten():
      self.assertGreaterEqual(r, 0)

    r1_returned, r2_returned = rank_returned[0, :], rank_returned[1, :]
    self.assertCountEqual(jnp.array([1, 2]), r1_returned[:2])
    self.assertEqual(0, r1_returned[2])
    self.assertCountEqual(jnp.array([3, 4, 7]), r1_returned[3:6])
    self.assertCountEqual(jnp.array([5, 6]), r1_returned[6:])

    self.assertEqual(7, r2_returned[0])
    self.assertEqual(1, r2_returned[1])
    self.assertCountEqual(jnp.array([3, 4]), r2_returned[2:4])
    self.assertEqual(2, r2_returned[4])
    self.assertCountEqual(jnp.array([0, 5, 6]), r2_returned[5:])

  def test_sample_and_sample_from_unranked_classes(self):
    gibbs_sampler = self.gibbs_samplers["jit_per_reader"]
    selectors_1 = [[1, 2], [3, 4], [17]]
    selectors_2 = [[1], [2, 3], [17]]
    selectors = [selectors_1, selectors_2]
    num_classes = 30
    shape_lam = jnp.ones((num_classes,)) * (1 / num_classes)
    rate_lam = jnp.ones((num_classes,))
    key = jax.random.PRNGKey(0)
    num_iterations = 10
    num_warm_up_iterations = 5
    results_sampler = gibbs_sampler.sample(
        key, selectors, shape_lam, rate_lam, num_iterations
    )
    results_sampler_unranked_0 = gibbs_sampler.sample_from_ranked_classes(
        key,
        selectors,
        1,
        num_classes,
        num_classes,
        num_iterations=num_iterations,
        represent_unranked_classes=False,
    )
    results_sampler_unranked_1_eq = gibbs_sampler.sample_from_ranked_classes(
        key,
        selectors,
        1,
        num_classes,
        num_classes,
        num_iterations=num_iterations,
        represent_unranked_classes=True,
        normalize_unranked_equally=True,
    )
    results_sampler_unranked_1_uneq = gibbs_sampler.sample_from_ranked_classes(
        key,
        selectors,
        1,
        num_classes,
        num_classes,
        num_iterations=num_iterations,
        represent_unranked_classes=True,
        normalize_unranked_equally=False,
    )
    self.assertEqual(
        jnp.sum(results_sampler_unranked_0[:, jnp.array([5, 12, 29])]), 0
    )
    # Checking whether all unranked classes are assigned the same value.
    values = np.concatenate(
        [
            results_sampler_unranked_1_eq[:, jnp.array([i])]
            for i in range(num_classes)
            if i not in jax.tree_util.tree_leaves(selectors)
        ]
    )
    np.testing.assert_array_almost_equal(
        values - values[0, 0], np.zeros_like(values), decimal=3
    )
    for i in range(num_iterations):
      self.assertNotEqual(
          results_sampler_unranked_1_uneq[i, 10],
          results_sampler_unranked_1_uneq[i, 20],
      )
    for results in [
        results_sampler,
        results_sampler_unranked_0,
        results_sampler_unranked_1_eq,
        results_sampler_unranked_1_uneq,
    ]:
      np.testing.assert_array_equal(
          results.shape, (num_iterations, num_classes)
      )
      avg_lambda = results[num_warm_up_iterations:].mean(axis=0)
      # Higher ranked classes must have higher lambdas on average.
      self.assertGreater(avg_lambda[1], avg_lambda[2])
      self.assertGreater(avg_lambda[1], avg_lambda[3])
      self.assertGreater(avg_lambda[1], avg_lambda[4])
      self.assertGreater(avg_lambda[1], avg_lambda[17])
      self.assertGreater(avg_lambda[2], avg_lambda[17])
      for i in [1, 2, 3, 4]:
        for j in range(5, 30):
          if j != 17:
            self.assertGreater(avg_lambda[i], avg_lambda[j])

  def test_gibbs_sampler_pl_nonnegative_nan(self):
    selector = [[[33], [45]], [[33], [45, 20, 31]], [[33], [45]], [[33]]]
    num_classes = 50
    shape_lam = jnp.ones((num_classes,)) / num_classes
    rate_lam = jnp.ones((num_classes,))
    key = jax.random.PRNGKey(0)
    num_iterations = 3
    results = {}
    jit_strategies = ["jit_per_reader", "jit_per_iteration", "no_jit"]
    for jit_strategy in jit_strategies:
      gibbs_sampler = self.gibbs_samplers[jit_strategy]
      results[jit_strategy] = gibbs_sampler.sample(
          key, selector, shape_lam, rate_lam, num_iterations=num_iterations
      )
      self.assertFalse(jnp.any(jnp.isnan(results[jit_strategy])))
      self.assertTrue(jnp.all(results[jit_strategy] > 0))
    for i in range(1, len(jit_strategies)):
      np.testing.assert_array_almost_equal(
          results[jit_strategies[i]], results[jit_strategies[i - 1]], decimal=4
      )


if __name__ == "__main__":
  absltest.main()
