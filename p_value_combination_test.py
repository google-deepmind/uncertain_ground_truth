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

"""Test for p-value combination utilities."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

import p_value_combination


class PValueCombinationTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(num_variables=10, num_dependent_variables=0),
      dict(num_variables=10, num_dependent_variables=10),
  ])
  def test_estimate_effective_number_of_tests(
      self, num_variables, num_dependent_variables
  ):
    ranks = jax.random.uniform(jax.random.PRNGKey(0), (10000, num_variables))
    if num_dependent_variables > 0:
      ranks = jnp.concatenate(
          (ranks, ranks[:, :num_dependent_variables]), axis=1
      )
    for method in ['cn', 'lj', 'gao', 'gal']:
      num_tests = jnp.round(
          p_value_combination.estimate_effective_number_of_tests(ranks, method)
      )
      if num_dependent_variables > 0:
        self.assertGreaterEqual(num_tests, num_variables)
        self.assertGreaterEqual(
            num_variables + num_dependent_variables, num_tests
        )
      else:
        self.assertEqual(jnp.round(num_tests), num_variables)

  @parameterized.parameters([
      dict(method='fisher'),
      dict(method='inverse_fisher'),
      dict(method='stouffer'),
      dict(method='tippett'),
      dict(method='bonferroni'),
  ])
  def test_combine_independent_p_values(self, method):
    for num_variables in [1, 10]:
      p_values = jax.random.uniform(
          jax.random.PRNGKey(0), (10000, num_variables)
      )
      combined_p_values = p_value_combination.combine_independent_p_values(
          p_values, num_variables, method
      )
      for u in [i / 10.0 for i in range(1, 10)]:
        if method != 'bonferroni':
          # Bonferroni does not result in a uniform combined p-value.
          self.assertAlmostEqual(jnp.mean(combined_p_values < u), u, places=1)
      for alpha in [0.05, 0.1]:
        self.assertTrue(
            jnp.allclose(jnp.mean(combined_p_values > alpha),
                         1 - alpha,
                         atol=0.01)
        )

  def test_combine_ecdf_p_values(self):
    num_examples = 100000
    # Test with uneven split to be sure that ranks are computed correctly.
    split = num_examples // 4
    p_values = jax.random.uniform(jax.random.PRNGKey(0), (num_examples, 10))
    p_values = jnp.concatenate((p_values, p_values), axis=1)
    combined_p_values = jnp.mean(p_values, axis=1)
    corrected_p_values = p_value_combination.combine_ecdf_p_values(
        combined_p_values[:split], combined_p_values[split:]
    )
    for u in [i / 10.0 for i in range(1, 10)]:
      self.assertAlmostEqual(jnp.mean(corrected_p_values < u), u, places=1)
    for alpha in [0.05, 0.1]:
      self.assertAlmostEqual(
          jnp.mean(corrected_p_values > alpha), 1 - alpha, places=2
      )


if __name__ == '__main__':
  absltest.main()
