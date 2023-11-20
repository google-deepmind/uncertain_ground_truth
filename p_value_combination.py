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

"""Working with p-values for conformal prediction.

References:
  [1] Ozan Cinar, Wolfgang Viechtbauer:
      The poolr Package for Combining Independent and Dependent p Values.
      J. Stat. Softw. 101(1) (2022).
"""

from typing import Optional

import jax
import jax.numpy as jnp
import scipy.stats


def estimate_effective_number_of_tests(
    ranks: jnp.ndarray, method: str
) -> jnp.ndarray:
  """Estimates the effective number of tests from computed ranks.

  Supported methods: `cn`, `lj`, `gao`, `gal`, see [1].
  Uses ranks as used for p-value computation and obtained by
  `conformal_prediction.compute_ranks`.

  Args:
    ranks: `num_samples x num_examples` matrix of ranks for the test conformity
      scores with respect to the validation conformity scores.
    method: One of the methods listed above.

  Returns:
    Effective number of tests based on selected method.
  """

  def eigenvalues_from_ranks(ranks: jnp.ndarray) -> jnp.ndarray:
    """Compute eigenvalues of the correlation matrix of ranks."""
    eigenvalues, _ = jax.numpy.linalg.eig(jnp.corrcoef(ranks.T))
    return eigenvalues

  def cn_estimate(ranks: jnp.ndarray) -> jnp.ndarray:
    """Follows Cheverud and Nyholt, Equation (5) in [1]."""
    eigenvalues = eigenvalues_from_ranks(ranks)
    num_samples = eigenvalues.shape[0]
    return 1 + (num_samples - 1) * (1 - jnp.var(eigenvalues) / num_samples)

  def lj_estimate(ranks: jnp.ndarray) -> jnp.ndarray:
    """Follows Li and Ji, Equation (6) in [1]."""
    eigenvalues = eigenvalues_from_ranks(ranks)
    eigenvalues = jnp.abs(eigenvalues)
    indicators = (eigenvalues >= 1).astype(float)
    indicators += eigenvalues - jnp.floor(eigenvalues)
    return jnp.sum(indicators)

  def gao_estimate(
      ranks: jnp.ndarray, c: Optional[float] = 0.995
  ) -> jnp.ndarray:
    """Follows Gao et al., Equation (7) in [1]."""
    eigenvalues = eigenvalues_from_ranks(ranks)
    cumulative_eigenvalues = jnp.cumsum(eigenvalues)
    denominator = jnp.sum(eigenvalues)
    return jnp.argmax((cumulative_eigenvalues / denominator) > c) + 1

  def gal_estimate(ranks: jnp.ndarray) -> jnp.ndarray:
    """Follows Galwey et al., Equation (8) in [1]."""
    eigenvalues = eigenvalues_from_ranks(ranks)
    eigenvalues = jnp.maximum(0, eigenvalues)
    return jnp.real(jnp.sum(jnp.sqrt(eigenvalues)) ** 2 / jnp.sum(eigenvalues))

  methods = {
      'cn': cn_estimate,
      'lj': lj_estimate,
      'gao': gao_estimate,
      'gal': gal_estimate,
  }
  if method not in methods:
    raise ValueError(
        f'Method {method} not supported, choose cn, lj, gao, or gal.'
    )
  return methods[method](ranks)


def combine_independent_p_values(
    p_values: jnp.ndarray, num_tests: int, method: str, axis: int = -1
) -> jnp.ndarray:
  """Combine independent p-values along a specific axis using various methods.

  Implementing the following methods to combine independent p-values,
  following [1]:
  * `fisher`: Fisher's method
  * `stouffer`: Stouffer's method
  * `inverse_fisher`: inverse Fisher method
  * `bonferroni`: Bonferroni's method
  * `tippett`: Tippett's method

  Here, `num_tests` is the number of independent tests to assume and can
  be different from the actual number of p-values passed.

  Args:
    p_values: p-values to combine along `axis`; for example, a `num_p_values x
      num_examples` shape array where `num_p_values` can be different from
      `num_tests`.
    num_tests: Number of independent tests to assume.
    method: The method to use, see supported conditions above.
    axis: Axis in `p_values` to combine p-values along.

  Returns:
    Combined p-values as `num_examples` shaped array.
  """

  def fishers_method(
      p_values: jnp.ndarray, num_tests: int, effective_tests: int, axis: int
  ) -> jnp.ndarray:
    """Fisher's chi-square method."""
    combined_p_values = -2 * jnp.sum(jnp.log(p_values), axis=axis)
    combined_p_values *= effective_tests / num_tests
    chi2 = scipy.stats.chi2(2 * num_tests)
    return 1 - jnp.array(chi2.cdf(combined_p_values))

  def bonferroni_method(
      p_values: jnp.ndarray,
      num_tests: int,  # pylint: disable=unused-argument
      effective_tests: int,
      axis: int,
  ) -> jnp.ndarray:
    """Bonferroni's method, taking the minimum p-value."""
    return jnp.minimum(1, jnp.min(p_values, axis=axis) * effective_tests)

  def tippetts_method(
      p_values: jnp.ndarray,
      num_tests: int,  # pylint: disable=unused-argument
      effective_tests: int,
      axis: int,
  ) -> jnp.ndarray:
    """Tippett's method, taking tha exponentiated minimum p-value."""
    return 1 - (1 - jnp.min(p_values, axis=axis)) ** effective_tests

  def stouffers_method(
      p_values: jnp.ndarray, num_tests: int, effective_tests: int, axis: int
  ) -> jnp.ndarray:
    """Stouffer's method using the inverse normal CDF."""
    z_values = jax.scipy.stats.norm.ppf(1 - p_values)
    z_values *= jnp.sqrt(effective_tests / num_tests)
    combined_z_values = jnp.sum(z_values, axis=axis) / jnp.sqrt(num_tests)
    return 1 - jax.scipy.stats.norm.cdf(combined_z_values)

  def inverse_fishers_method(
      p_values: jnp.ndarray, num_tests: int, effective_tests: int, axis: int
  ) -> jnp.ndarray:
    """Inverse Fisher method, using the inverse chi-square CDF."""
    chi2_one = scipy.stats.chi2(1)
    combined_z_values = jnp.sum(chi2_one.ppf(1 - p_values), axis=axis)
    combined_z_values *= effective_tests / num_tests
    chi2_num_tests = scipy.stats.chi2(num_tests)
    return 1 - jnp.array(chi2_num_tests.cdf(combined_z_values))

  num_p_values = p_values.shape[axis]
  methods = {
      'fisher': fishers_method,
      'stouffer': stouffers_method,
      'inverse_fisher': inverse_fishers_method,
      'bonferroni': bonferroni_method,
      'tippett': tippetts_method,
  }
  if method not in methods:
    raise ValueError(
        f'Method {method} not supported, choose '
        'fisher, stouffer, inverse_fisher, bonferroni or tippett.'
    )
  return methods[method](p_values, num_p_values, num_tests, axis)


def combine_ecdf_p_values(
    val_p_values: jnp.ndarray, test_p_values: jnp.ndarray
) -> jnp.ndarray:
  """Combine p-values using the empirical cumulative distribution function.

  Does not assume any dependence or independence of the p-values but does
  require a separate set of held-out p-values to "learn" the inverse CDF on.

  In the conformal prediction context, `val_p_values` need to be the
  p-values corresponding to the true labels (otherwise, the p-values are not
  valid p-values). Then, `test_p_values` can be a flattened version of
  p-values for all possible labels.

  Args:
    val_p_values: `num_val_examples` shaped array of combined validation
      p-values representing the cumulative distribution function.
    test_p_values: `num_test_examples` shaped array of combined test p-values.

  Returns:
    Combined p-values
  """
  test_ranks = jnp.expand_dims(val_p_values, axis=0) <= jnp.expand_dims(
      test_p_values, axis=1
  )
  test_ranks = jnp.sum(test_ranks, axis=1)
  test_corrected_p_values = test_ranks / val_p_values.shape[0]
  return test_corrected_p_values
