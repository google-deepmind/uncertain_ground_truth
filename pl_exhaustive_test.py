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

"""Tests for pl_exhaustive."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

import pl_exhaustive


class PlackettLuceExhaustiveTest(parameterized.TestCase):

  def get_test_data_pl_likelihood(self):
    """Test data for PL likelihood functions.

    Returns:
      The dictionary for variables needed for testing the likelihood functions.
    """
    theta_1 = jnp.array([0.3, 0.15, 0.4, 0.1, 0.05])
    theta_2 = jnp.array([0.1, 0.3, 0.1, 0.15, 0.35])
    phi = jnp.vstack((jnp.log(theta_1) + 1.0, jnp.log(theta_2) + 1.0))
    num_readers = 2
    batch_size, num_classes = phi.shape
    rankings = jnp.zeros((batch_size, num_readers, num_classes), dtype=int)
    # The first instance has two readers.
    rankings = rankings.at[0, 0, :].set(jnp.array([3, 0, 1, 2, 4]))
    rankings = rankings.at[0, 1, :].set(jnp.array([2, 1, 4, 3, 0]))
    # The second instance has one reader.
    rankings = rankings.at[1, 0, :].set(jnp.array([1, 4, 3, 0, 2]))
    selectors_1_0 = [[3, 0], [1, 2]]
    selectors_1_1 = [[2], [1], [4]]
    selectors_1_2 = [[3, 0], [1, 2], [4]]  # Equivalent to selectors_1_0.
    selectors_2 = [[1, 4], [3]]
    selectors = [[selectors_1_0, selectors_1_1, selectors_1_2], [selectors_2]]
    likelihood_1_0 = ((0.1 * (0.3 / 0.9)) + (0.3 * (0.1 / 0.7))) * (
        ((0.15 / 0.6) * (0.4 / 0.45)) + ((0.4 / 0.6) * (0.15 / 0.2))
    )
    likelihood_1_1 = (0.4) * (0.15 / 0.6) * (0.05 / 0.45)
    likelihood_1_2 = likelihood_1_0
    likelihood_1 = likelihood_1_0 * likelihood_1_1 * likelihood_1_2
    likelihood_2 = ((0.3 * (0.35 / 0.7)) + (0.35 * (0.3 / 0.65))) * (
        0.15 / 0.35
    )
    correct_loglikelihoods = jnp.log(jnp.array([likelihood_1, likelihood_2]))
    correct_reader_loglikelihoods = [
        jnp.log(jnp.array([likelihood_1_0, likelihood_1_1, likelihood_1_2])),
        jnp.log(jnp.array([likelihood_2])),
    ]
    return {  # pytype: disable=bad-return-type  # jax-ndarray
        "phi": phi,
        "rankings": rankings,
        "selectors": selectors,
        "correct_loglikelihoods": correct_loglikelihoods,
        "correct_reader_loglikelihoods": correct_reader_loglikelihoods,
    }

  def test_full_top_k_ordering_given_sum(self):
    ordered_lam = jnp.array([3., 2., 4.])
    sum_lam = 12.
    ordered_phi = jnp.log(ordered_lam)
    logsumexp_phi = jnp.log(sum_lam)

    ll_expected = jnp.log((3 / 12) * (2 / 9) * (4 / 7))

    var_full_top_k_ordering_given_sum = jax.jit(
        pl_exhaustive.full_top_k_ordering_given_sum
    )
    ll_returned = var_full_top_k_ordering_given_sum(ordered_phi, logsumexp_phi)
    self.assertAlmostEqual(ll_expected, ll_returned, places=5)

  def test_pl_loglikelihood_group(self):
    lam = jnp.array([1., 0.5, 3.])
    sum_lam = 6.
    logsumexp_phi = jnp.log(sum_lam)
    selector = [1, 2]
    phi = jnp.log(lam)

    ll_expected = jnp.log((.5 / 6) * (3 / 5.5) + (3 / 6) * (.5 / 3))
    var_pl_loglikelihood_group = jax.jit(pl_exhaustive._pl_loglikelihood_group)
    ll_returned = var_pl_loglikelihood_group(phi, logsumexp_phi, selector)
    self.assertAlmostEqual(ll_expected, ll_returned)

  def test_pl_loglikelihood_reader(self):
    test_data = self.get_test_data_pl_likelihood()
    example_indices = [0, 1]
    reader_indices = [[0, 1, 2], [0]]
    var_pl_loglikelihood_reader = jax.jit(
        pl_exhaustive._pl_loglikelihood_reader
    )
    for example_index in example_indices:
      for reader_index in reader_indices[example_index]:
        ll_returned = var_pl_loglikelihood_reader(
            test_data["phi"][example_index],
            test_data["selectors"][example_index][reader_index],
            )
        self.assertAlmostEqual(
            test_data["correct_reader_loglikelihoods"][example_index]
            [reader_index],
            ll_returned,
            places=5)

  def test_pl_loglikelihood_reader_full_partial_ranking_equivalence(self):
    phi = jnp.log(jnp.array([0.3, 0.15, 0.4, 0.1, 0.05])) + 1.0
    selectors_1 = [[3, 0], [1, 2]]
    selectors_2 = [[3, 0], [1, 2], [4]]
    ll_1 = pl_exhaustive._pl_loglikelihood_reader(
        phi,
        selectors_1,
    )
    ll_2 = pl_exhaustive._pl_loglikelihood_reader(
        phi,
        selectors_2,
    )
    self.assertEqual(
        ll_1,
        ll_2,
    )

  def test_pl_loglikelihood_single(self):
    test_data = self.get_test_data_pl_likelihood()
    example_index = 0
    var_pl_loglikelihood_single = jax.jit(pl_exhaustive.pl_loglikelihood_single)
    ll_returned = var_pl_loglikelihood_single(
        test_data["phi"][example_index],
        test_data["selectors"][example_index])
    self.assertAlmostEqual(
        test_data["correct_loglikelihoods"][example_index],
        ll_returned,
        places=5)
    # A test case specific to the exhaustive method. In the exhaustive method.
    theta = jnp.array([0.15, 0.4, 0.05, 0.05, 0.35])
    phi = jnp.log(theta) + 1.5
    selector_1 = [[[4, 2], [3]]]
    # Adding the last group should not change behavior.
    selector_2 = [[[4, 2], [3], [0, 1]]]
    ll_returned_1 = var_pl_loglikelihood_single(
        phi, selector_1)
    ll_returned_2 = var_pl_loglikelihood_single(
        phi, selector_2)
    self.assertAlmostEqual(ll_returned_1, ll_returned_2, places=5)

  def test_pl_loglikelihood_batch(self):
    test_data = self.get_test_data_pl_likelihood()
    for var_pl_loglikelihood_batch in [
        pl_exhaustive.pl_loglikelihood_batch_jit_per_instance,
        pl_exhaustive.pl_loglikelihood_batch_jit_per_reader,
        pl_exhaustive.pl_loglikelihood_batch,
        jax.jit(pl_exhaustive.pl_loglikelihood_batch),
    ]:
      ll_returned = var_pl_loglikelihood_batch(
          test_data["phi"], test_data["selectors"])
      np.testing.assert_array_almost_equal(
          test_data["correct_loglikelihoods"], ll_returned
      )


if __name__ == "__main__":
  absltest.main()
