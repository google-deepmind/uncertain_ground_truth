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

"""Computing Plackett-Luce loglikelihood given multiple partial rankings using the exhaustive approach.

Plackett-Luce model is a distribution on permutations based on sampling
without replacement.

If there are J items, the first item s(1) is selected with probability
proportional to lambda_j where
Pr[s(1) = j] = lambda_j/Z

with Z = lambda_1 + lambda_2 + .. + lambda_J

Then, the second item is chosen by excluding the first item with probability

Pr[s(2) = j | s(1)] = [j != s(1)] * lambda_j/(Z - lambda_{s(1)})

and so forth.

A differential diagnosis (ddx) is a collection of partial rankings provided by
several readers by ranking several conditions.

Terminology:

Even though the concept "partial ranking" actually subsumes any tied ranking,
including an untied top-k ranking (since the remaining L-k items can be
considered to be tied), we will be overloading this term with the following
definitions for the sake of convenience:

Full ranking: An untied ranking of all L options.
Partial ranking: A ranking of all L options that contains ties.
Full top-k ranking: A top-k ranking that does not contain ties.
Partial top-k ranking: A top-k ranking that contains ties within the top-k.

Within the code, normalized probabilities are denoted by `theta`, unnormalized
probabilities are denoted by `lam` (standing for lambda), and normalized or
unnormalized log-probabilities are denoted by `phi`.

This module includes the implementation of an exhaustive approach.

References:
- Caron2012: Caron, François, Yee Whye Teh, and Thomas Brendan Murphy. 2012.
    “Bayesian Nonparametric Plackett-Luce Models for the Analysis of Preferences
    for College Degree Programmes.” arXiv [stat.ML]. arXiv.
"""

import itertools
from typing import List, NewType, Sequence

import jax
import jax.numpy as jnp
from jax.tree_util import Partial
from typing_extensions import Protocol

Selector = NewType("Selector", Sequence[int])


class ReaderLoglikelihoodFn(Protocol):
  """Type for a generic reader loglikelihood function."""

  def __call__(
      self, phi: jnp.ndarray, selectors: List[Selector]
  ) -> jnp.ndarray:
    """Generic function for computing reader log likelihood.

    Args:
      phi: num_classes sized vector of logplausibilities.
      selectors: Selectors for a given group.

    Returns:
      Loglikelihood for the given reader.
    """


class SingleLoglikelihoodFn(Protocol):
  """Type for a generic single instance loglikelihood function."""

  def __call__(
      self, phi: jnp.ndarray, selectors: List[List[Selector]]
  ) -> jnp.ndarray:
    """Generic function for computing instance loglikelihood.

    Args:
      phi: num_classes sized vector of logplausibilities.
      selectors: Indices of the classes in the group.

    Returns:
      Loglikelihood for the given instance.
    """


class BatchLoglikelihoodFn(Protocol):
  """Type for a generic batch loglikelihood function."""

  def __call__(
      self, phi: jnp.ndarray, selectors: List[List[List[Selector]]]
  ) -> jnp.ndarray:
    """Generic function for computing batch loglikelihood.

    Args:
      phi: batch_size x num_classes sized vector of logplausibilities.
      selectors: Indices of the classes in the group.

    Returns:
      Loglikelihood for the given batch.
    """


def full_top_k_ordering_given_sum(
    ordered_phi: jnp.ndarray, logsumexp_phi: jnp.ndarray
) -> jnp.ndarray:
  """PL loglikelihood of a k-sized ordering within a larger L-sized ordering.

  Given L options, let a rho = (rho_1, rho_2, ...) be an ordering that does not
  contain ties within it. This method computes the loglikelihood of this
  ordering given the logsumexp of log-plausibilities of all items ranked as high
  as rho_1 or lower. See below for an example.

  For the sake of convenience, let us follow this example through plausibilities
  rather than logplausibilities. For a given logplausibility
  phi_i, let lambda_i = exp(phi_i). Let L = 10 and imagine a vector of ranked
  plausibilities (lambda_1, ..., lambda_10), that is, plausibilities associated
  with (rho_1, ..., rho_10). Let E_i be the sum of plausibilities for items
  ranked i or lower. This method, for example, accepts
  S = (lambda_3, lambda_4, lambda_5) and E_3. It returns log((lambda_3/E_3) *
  (lambda_4/(E_3-lambda_3) * (lambda_5/(E_3-e_3-e_4)).

  Args:
    ordered_phi: 1-dim array of a desired order of log-plausibilities
    logsumexp_phi: The logsumexp of the log-plausibilities of every option that
      is ranked equal to or lower than the highest ranked item in the current
      set.

  Returns:
    PL loglikelihood of the given ranking.
  """
  num_elements = len(ordered_phi)
  ordered_phi_repeated = jnp.tile(ordered_phi[None, :], reps=(num_elements, 1))
  mask = jnp.flip(jnp.triu(m=jnp.ones((num_elements, num_elements))), 1)
  denoms = jnp.flip(
      jax.nn.logsumexp(
          ordered_phi_repeated,
          axis=1,
          b=mask,
      )
  )
  mask = jnp.vstack((jnp.ones(num_elements), -jnp.ones(num_elements)))
  sum_denoms = jnp.vstack((jnp.repeat(logsumexp_phi, num_elements), denoms))
  denoms = jax.nn.logsumexp(sum_denoms, axis=0, b=mask)
  denoms = jnp.concatenate((jnp.array([logsumexp_phi]), denoms))[:-1]
  log_probs = ordered_phi - denoms
  log_prob = jnp.sum(log_probs, axis=0)
  return log_prob


def _pl_loglikelihood_group(
    phi: jnp.ndarray, logsumexp_phi: jnp.ndarray, selector: Selector
) -> jnp.ndarray:
  """PL loglikelihood of a given tied group within a larger ranking.

  For a given k-sized tied group within a larger set of L items, this method
  computes the log-likelihoods for the k! possible orderings of this set
  and logsumexp's over these. See methods called for log-likelihood evaluation
  for more details.

  Args:
    phi: Log-plausibilities of all L items (ordered according to item index, not
      according to a particular ranking by a reader).
    logsumexp_phi: Logsumexp of log-plausibilities of all items that are ranked
      as high as or lower than the current group.
    selector: The indices for the elements in this specific group.

  Returns:
    PL loglikelihood of the tied group given the larger ranking.
  """
  ordered_phi = phi[jnp.array(selector, dtype=int)]
  ordered_phi_perms = jnp.vstack(
      [jnp.asarray(v) for v in itertools.permutations(ordered_phi)]
  )
  log_prob = jax.nn.logsumexp(
      jax.vmap(full_top_k_ordering_given_sum, in_axes=[0, None], out_axes=0)(
          ordered_phi_perms, logsumexp_phi
      )
  )
  return log_prob


def _pl_loglikelihood_single(
    phi: jnp.ndarray,
    selectors: List[List[Selector]],
    reader_loglikelihood_fn: ReaderLoglikelihoodFn,
) -> jnp.ndarray:
  """Brute force Plackett-Luce loglikelihood for a partial ordering of k options.

  The function accepts a partial top-k ordering and the log-plausibilities of
  all L options. For reader r the ordering can be thought of as a tuple of sets
  (B_{r,1}, B_{r,2}, ...), where the sum of cardinalities of all sets are k.
  This function sums over the loglikelihoods associated with each reader for
  a single case.

  Args:
    phi: num_classes-sized array of log-plausibilities
    selectors: A nested list with selectors as leaves, where each selector
      corresponds to the s'th set B_{r,s} by reader r, with the contents of the
      list being indices of the members of B_{r,s}.
    reader_loglikelihood_fn: Function that computes per-reader likelihood.

  Returns:
    Plackett-Luce loglikelihood of the log-plausibilities for the given
      rankings.
  """
  return sum(
      reader_loglikelihood_fn(phi, reader_selectors)
      for reader_selectors in selectors
  )


def _pl_loglikelihood_reader(
    phi: jnp.ndarray,
    selectors: List[Selector],
) -> jnp.ndarray:
  """Brute force PL loglikelihood for a partial ordering of k options for a single ranking.

  The function accepts a partial top-k ordering and the log-plausibilities of
  all L options. For reader r the ordering can be thought of as a tuple of sets
  (B_{r,1}, B_{r,2}, ...), where the sum of cardinalities of all sets are k.
  This function computes the log-likelihood for a single ranking or reader.

  This method uses a brute force approach to compute the likelihood of the given
  parameters with respect to the observed ranking. However, suppressing the
  index r, the method exploits the independence of P(B_{s}) and P(B_{s}|B_{s-1},
  B_{s-2}, ...), thus has a complexity of O(N+S!) where S is the size of the
  largest B_{s}.

  Args:
    phi: num_classes-sized array of log-plausibilities
    selectors: A list with selectors as components, where each selector
      corresponds to the s'th set B_{r,s} by the current reader r, with the
      contents of the list being indices of the members of B_{r,s}.

  Returns:
    Plackett-Luce loglikelihood of the log-plausibilities for the given ranking
      by a single reader.
  """
  log_prob = 0
  for selector in selectors:
    selector = jnp.array(selector, dtype=int)
    log_prob += _pl_loglikelihood_group(phi, jax.nn.logsumexp(phi), selector)  # pytype: disable=wrong-arg-types  # jnp-type
    # Logplausibilities of the classes already seen are set to -inf so that
    # they have no contribution to next group's denominator computation.
    phi = phi.at[selector].set(-jnp.inf)
  return log_prob  # pytype: disable=bad-return-type  # jax-ndarray


def _pl_loglikelihood_batch(
    phi: jnp.ndarray,
    selectors: List[List[List[Selector]]],
    single_loglikelihood_fn: SingleLoglikelihoodFn,
) -> jnp.ndarray:
  """Brute force Plackett-Luce loglikelihoods for a batch of partial orderings.

  This method computes the brute force loglikelihoods for a batch of
  log-plausibilities and partial orderings. See the likelihood function called
  within the for loop for more information regarding the computation approach.

  Args:
    phi: batch_size x num_classes-sized array of log-plausibilities
    selectors: A nested list of selectors where each selector corresponds to
      s'th set B_{i,r,s} by reader r for instance i, with the contents of the
      list being indices of the members of B_{i,r,s}.
    single_loglikelihood_fn: Function that computes per-instance likelihood.

  Returns:
    Plackett-Luce loglikelihoods of the log-plausibilities for the given
      rankings.
  """
  return jnp.stack(
      [
          single_loglikelihood_fn(phi_i, selectors_i)
          for phi_i, selectors_i in zip(phi, selectors)
      ]
  )


pl_loglikelihood_single = Partial(
    _pl_loglikelihood_single, reader_loglikelihood_fn=_pl_loglikelihood_reader
)
pl_loglikelihood_single_jit_per_reader = Partial(
    _pl_loglikelihood_single,
    reader_loglikelihood_fn=jax.jit(_pl_loglikelihood_reader),
)
pl_loglikelihood_single_jit_per_instance = jax.jit(
    pl_loglikelihood_single_jit_per_reader
)
pl_loglikelihood_batch = Partial(
    _pl_loglikelihood_batch, single_loglikelihood_fn=pl_loglikelihood_single
)
pl_loglikelihood_batch_jit_per_reader = Partial(
    _pl_loglikelihood_batch,
    single_loglikelihood_fn=pl_loglikelihood_single_jit_per_reader,
)
pl_loglikelihood_batch_jit_per_instance = Partial(
    _pl_loglikelihood_batch,
    single_loglikelihood_fn=pl_loglikelihood_single_jit_per_instance,
)
