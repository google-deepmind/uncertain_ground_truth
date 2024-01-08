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

"""Aggregating partial rankings using the Plackett-Luce choice model.

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

References:
- Caron2012: Caron, François, Yee Whye Teh, and Thomas Brendan Murphy. 2012.
    “Bayesian Nonparametric Plackett-Luce Models for the Analysis of Preferences
    for College Degree Programmes.” arXiv [stat.ML]. arXiv.
    http://arxiv.org/abs/1211.5037.
- Kool2020: Kool, Wouter, Herke van Hoof, and Max Welling. 2020. “Estimating
    Gradients for Discrete Random Variables by Sampling without Replacement.”
    https://openreview.net/pdf?id=rklEj2EFvB.
- Ma2021: Ma, Jiaqi, Xinyang Yi, Weijing Tang, Zhe Zhao, Lichan Hong, Ed Chi,
    and Qiaozhu Mei. 13--15 Apr 2021. “Learning-to-Rank with Partitioned
    Preference: Fast Estimation for the Plackett-Luce Model.” In Proceedings of
    The 24th International Conference on Artificial Intelligence and Statistics,
    edited by Arindam Banerjee and Kenji Fukumizu, 130:928–36. Proceedings of
    Machine Learning Research. PMLR.
"""

import itertools
import logging
from typing import List, Tuple

import jax
import jax.numpy as jnp

import pl_exhaustive


class GibbsSamplerPlackettLuce:
  """Gibbs sampler class for Plackett-Luce distribution.

  See the component functions for the details of the methodology.
  """

  def __init__(self, jit_strategy: str):
    """Initialize the Gibbs sampler by determining the jitting strategy.

    Args:
      jit_strategy: Jitting strategy for the sampler. Three implemented options
        are jit_per_reader, jit_per_iteration, no_jit.

    Raises:
      ValueError: If the jitting strategy provided is not available.
    """
    self.jit_strategy = jit_strategy
    self._vmapped_sample_tau_given_lam_and_rankings = jax.vmap(
        self._sample_tau_given_lam_and_rankings, in_axes=[0, None, 0]
    )
    if jit_strategy == "jit_per_reader":
      self._vmapped_sample_tau_given_lam_and_rankings = jax.jit(
          self._vmapped_sample_tau_given_lam_and_rankings
      )
      self._sample_lam_given_tau = jax.jit(self._sample_lam_given_tau)
      self._per_reader_sample_perm_given_lam_and_partial_rankings = jax.jit(
          self._per_reader_sample_perm_given_lam_and_partial_rankings
      )
    elif jit_strategy == "jit_per_iteration":
      self._gibbs_sampler_pl_iteration = jax.jit(
          self._gibbs_sampler_pl_iteration
      )
    elif jit_strategy == "no_jit":
      pass
    else:
      raise ValueError(
          f'The jitting strategy entered "{jit_strategy}" is not valid. Valid '
          'options are: "jit_per_reader", "jit_per_iteration", "no_jit".'
      )

  def sample(
      self,
      key: jnp.ndarray,
      selectors: List[List[pl_exhaustive.Selector]],
      shape_lam: jnp.ndarray,
      rate_lam: jnp.ndarray,
      num_iterations: int = 100,
  ) -> jnp.ndarray:
    """Gibbs sampler for Plackett-Luce distribution on multiple partial rankings.

    This function conducts Gibbs sampling based on the Plackett-Luce model for
    multiple partial rankings by sampling arrival times (tau), plausibilities
    (lambda), and full rankings (sigma) from their full conditionals, based on
    the samples from the previous iteration.

    Args:
      key: PRNG key
      selectors: A nested list of selectors where each selector stands for a
        group of tied options for a reader, i.e. B_{r, m}
      shape_lam: num_classes sized array of shape hyperparameters for lambda
      rate_lam: num_classes sized array of rate hyperparameters for lambda
      num_iterations: Number of iterations for the Gibbs sampler

    Returns:
      results: num_iterations x num_classes sized array of plausibility samples.
    """
    num_classes = len(shape_lam)
    init_key, lam_key, perm_key, key = jax.random.split(key, 4)
    # Initialize the rankings array.
    rankings = self._initialize_rankings(init_key, selectors, num_classes)
    # Sample lambda_0 and rankings_0 to start the Gibbs iterations.
    lam = jax.random.gamma(lam_key, a=shape_lam) / rate_lam
    rankings = self._sample_perm_given_lam_and_partial_rankings(
        perm_key, rankings, selectors, lam
    )
    results = []
    for _ in range(num_iterations):
      key, lam, rankings = self._gibbs_sampler_pl_iteration(
          key, lam, rankings, selectors, shape_lam, rate_lam
      )
      results.append(lam)
    return jnp.stack(results)

  def sample_from_ranked_classes(
      self,
      key: jnp.ndarray,
      selectors: List[List[pl_exhaustive.Selector]],
      shape_lam_i: jnp.float64,
      rate_lam_i: jnp.float64,
      num_classes: int,
      num_iterations: int = 100,
      represent_unranked_classes: bool = True,
      normalize_unranked_equally: bool = False,
  ) -> jnp.ndarray:
    """PL Gibbs sampler that either ignores or collects unseen classes.

    Let number of seen rankings be K in a set of partial rankings. This wrapper
    function reduces the number of classes to K + a, where a = 1 if
    represent_unranked_classes is True and a = 0 otherwise, and return samples
    accordingly.

    Args:
      key: PRNG key.
      selectors: Selectors for single example with potentially multiple readers.
      shape_lam_i: The shape parameter that will be broadcast to K + a classes.
      rate_lam_i: The rate parameter that will be broadcast to K + a classes.
      num_classes: Total number of classes in the data.
      num_iterations: Number of desired Gibbs samples.
      represent_unranked_classes: If set to true, collects unranked under one
        class. Else, ignores the unranked classes completely.
      normalize_unranked_equally: If this is set to True and
        num_unranked_classes = 1, the remaining plausibilituies for each
        iteration are divided equally. Else, they are divided according to a
        Dirichlet(1) sample. The latter requires that shape_lam_i = 1.

    Returns:
      gibbs_samples: num_samples x (K + a) array of Gibbs samples.
      new_selectors: Selectors with the indexing classes changed to the one used
        in the sampling in this function.
      inv_mapping: The dictionary that includes the mapping from the new indices
        for observed classes to previously specified classes. Maps to -1 for all
        previously unseen classes if they exist.

    Raises:
      ValueError: If normalize_unranked_equally = False but shape_lam_i != 1.
    """
    if not normalize_unranked_equally and shape_lam_i != 1:
      raise ValueError(
          "normalize_unranked_equally = False requires shape_lam_i = 1."
      )
    ranked_classes = jnp.array(
        sorted(list(set(jax.tree_util.tree_leaves(selectors))))
    ).astype(int)
    num_ranked_classes = len(ranked_classes)
    if num_ranked_classes < 1:
      logging.warning("No ranked classes found.")
    num_classes_new = num_ranked_classes + int(represent_unranked_classes)
    shape_lam = jnp.ones((num_classes_new,)) * shape_lam_i
    rate_lam = jnp.ones((num_classes_new,)) * rate_lam_i
    mapping = {int(value): key for key, value in enumerate(ranked_classes)}
    new_selectors = jax.tree_util.tree_map(lambda x: mapping[x], selectors)
    lkey, key = jax.random.split(key)
    new_gibbs_samples = self.sample(
        key, new_selectors, shape_lam, rate_lam, num_iterations
    )
    gibbs_samples = jnp.zeros((num_iterations, num_classes))
    if represent_unranked_classes:
      unranked_classes = jnp.array(
          list(
              set(jnp.arange(num_classes, dtype=int).tolist())
              - set(ranked_classes.tolist())
          )
      ).astype(int)
      if normalize_unranked_equally:
        unranked_factor = 1 / len(unranked_classes)
      else:
        unranked_factor = jax.random.dirichlet(
            lkey, alpha=jnp.ones(len(unranked_classes)), shape=(num_iterations,)
        )
      idx_0 = jnp.tile(
          jnp.arange(num_iterations)[:, None], (1, len(unranked_classes))
      )
      idx_1 = jnp.tile(unranked_classes, (num_iterations, 1))
      gibbs_samples = gibbs_samples.at[idx_0, idx_1].set(
          new_gibbs_samples[:, [-1]] * unranked_factor
      )
      new_gibbs_samples = new_gibbs_samples[:, :-1]
    # Indices for writing samples to a num_iterations x num_classes sized array.
    idx_0 = jnp.tile(
        jnp.arange(num_iterations)[:, None], (1, num_ranked_classes)
    )
    idx_1 = jnp.tile(ranked_classes, (num_iterations, 1))
    gibbs_samples = gibbs_samples.at[idx_0, idx_1].set(new_gibbs_samples)
    return gibbs_samples

  def _sample_tau_given_lam_and_rankings(
      self, key: jnp.ndarray, lam: jnp.ndarray, rankings: jnp.ndarray
  ) -> jnp.ndarray:
    """Given a permutation and latent plausibilities, sample arrival times tau.

    This function samples from P(tau|lambda, sigma), where tau stands for the
    arrival times, lambda stands for plausibilities, and sigma stands for a full
    ranking. As demonstrated by Caron2012, a useful way of conducting this
    sampling is to sample the interarrival times from the appropriate
    exponential distribution and compute the cumulative sum of these to arrive
    at arrival times tau.

    Args:
      key: PRNG key
      lam: num_classes sized array of plausibilities
      rankings: num_classes sized array of rankings for each reader

    Returns:
      tau: num_classes sized array of arrival times for each option
    """
    normalizer_lam = jnp.sum(lam)
    rates = normalizer_lam - jnp.cumsum(
        jnp.concatenate([jnp.zeros((1,)), lam[rankings]])
    )
    rates = jnp.where(rates < 1e-6, 1e-6, rates)
    return jnp.cumsum(
        jax.random.exponential(key, shape=lam.shape) / rates[0:-1]
    )[jnp.argsort(rankings)]

  def _sample_lam_given_tau(
      self,
      key: jnp.ndarray,
      tau: jnp.ndarray,
      shape_lam: jnp.ndarray,
      rate_lam: jnp.ndarray,
  ) -> jnp.ndarray:
    """Given arrival times, sample from latent plausibilities.

    This function samples from P(lambda|tau), where tau stands for the
    arrival times and lambda stands for plausibilities. See Caron2012 for more
    details.

    Args:
      key: PRNG key
      tau: num_readers x num_classes sized array of arrival times for each
        reader and option
      shape_lam: num_classes sized array of shape hyperparameters for lambda
      rate_lam: num_classes sized array of rate hyperparameters for lambda

    Returns:
      lam: num_classes sized array of plausibilities for each option
    """
    shape_params = tau.shape[0] + shape_lam
    rate_params = jnp.sum(tau, axis=0) + rate_lam
    return jax.random.gamma(key, a=shape_params) / rate_params

  def _sample_from_block_posterior(
      self,
      key: jnp.ndarray,
      phi: jnp.ndarray,
      logsumexp_phi: jnp.ndarray,
      selector: pl_exhaustive.Selector,
  ) -> jnp.ndarray:
    """Sample from the posterior of a block given the lognormalizing constant.

    This function samples from the posterior of a block given
    log-plausibilities, block elements and the logsumexp of following groups.

    Args:
      key: PRNG key
      phi: num_classes sized array of logplausibilities
      logsumexp_phi: The logsumexp of the logplausibilities of the following
        groups
      selector: A selector for the given group

    Returns:
      A sample from the posterior of all permutations in the given group.
    """
    new_selectors = jnp.vstack(
        [jnp.asarray(v) for v in itertools.permutations(selector)]
    )
    ordered_phi_perms = phi[new_selectors]
    ordering_lls = jax.vmap(
        pl_exhaustive.full_top_k_ordering_given_sum,
        in_axes=[0, None],
        out_axes=0,
    )(ordered_phi_perms, logsumexp_phi)
    block_posterior = jnp.exp(ordering_lls - jax.nn.logsumexp(ordering_lls))
    new_selector = jax.random.choice(
        key=key,
        a=jnp.array(new_selectors, dtype=int),
        replace=False,
        p=block_posterior,
    )
    return new_selector

  def _get_denoms(self, ordered_phi: jnp.ndarray) -> jnp.ndarray:
    """Reverse logcumsumexp of a series of a given order of log-plausibilities.

    For an array of [phi_1, phi_2, phi_3], this method computes
    [lse(phi_1, phi_2, phi_3), lse(phi_2, phi_3), lse(phi_3)] where lse stands
    for logsumexp.

    Args:
      ordered_phi: 1-dim array of a desired order of log-plausibilities

    Returns:
      Reverse logcumsumexp of the given order of log-plausibilities.
    """
    num_classes = ordered_phi.shape[0]
    ordered_phi_rep = jnp.tile(ordered_phi[None, :], reps=(num_classes, 1))
    mask = jnp.triu(m=jnp.ones((num_classes, num_classes)))
    denoms = jax.nn.logsumexp(ordered_phi_rep, axis=1, b=mask, keepdims=False)
    return denoms

  def _per_reader_sample_perm_given_lam_and_partial_rankings(
      self,
      key: jnp.ndarray,
      phi: jnp.ndarray,
      rankings: jnp.ndarray,
      reader_selectors: List[pl_exhaustive.Selector],
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample per reader rankings for the Gibbs sampler.

    Args:
      key: PRNG key.
      phi: Logplausibilities of all classes.
      rankings: Rankings for the reader in question.
      reader_selectors: Selectors for the reader in question.

    Returns:
      key: PRNG key.
      new_rankings: Newly generated rankings.
    """
    ordered_phi = phi[rankings]
    new_rankings = jnp.zeros_like(rankings)
    denoms = self._get_denoms(ordered_phi)
    idx = 0
    for selector in reader_selectors:
      cur_group_size = len(selector)
      lkey, key = jax.random.split(key)
      new_selector = self._sample_from_block_posterior(  # pytype: disable=wrong-arg-types  # jnp-type
          lkey, phi, denoms[idx], jnp.array(selector, dtype=int)
      )
      new_rankings = new_rankings.at[idx : idx + cur_group_size].set(
          jnp.array(new_selector, dtype=int)
      )
      idx = idx + cur_group_size
    lkey, key = jax.random.split(key)
    probs = jnp.exp(phi[rankings[idx:]] - denoms[idx])
    last_ranks = jax.random.choice(
        key=lkey,
        a=rankings[idx:],
        shape=((len(rankings[idx:]),)),
        replace=False,
        p=probs,
    )
    new_rankings = new_rankings.at[idx:].set(last_ranks)
    return key, new_rankings

  def _sample_perm_given_lam_and_partial_rankings(
      self,
      key: jnp.ndarray,
      rankings: jnp.ndarray,
      selectors: List[List[pl_exhaustive.Selector]],
      lam: jnp.ndarray,
  ) -> jnp.ndarray:
    """Sample full rankings for each reader given lambda and partial rankings.

    This function samples from P(sigma|lambda, B) for each reader
    where sigma stands for a full ranking, lambda stands for plausibilities, and
    B_r = (B_{r,1}, B_{r,2}, ...) stands for a partial ranking for reader r,
    i.e. a tuple of sets where each set includes the indices of tied options.
    The posterior corresponds to the normalized likelihoods of each full
    ordering that conforms to the partial ordering specified.

    Args:
      key: PRNG key
      rankings: num_readers x num_classes sized array of rankings
      selectors: A nested list of selectors where each selector stands for a
        group of tied options for a reader, i.e. B_{r, m}
      lam: num_classes sized array of plausibilities

    Returns:
      rankings: num_readers x num_classes sized array of newly sampled rankings
    """
    phi = jnp.log(lam)
    for reader, reader_selectors in enumerate(selectors):
      key, new_reader_r = (
          self._per_reader_sample_perm_given_lam_and_partial_rankings(
              key, phi, rankings[reader], reader_selectors
          )
      )
      rankings = rankings.at[reader, :].set(new_reader_r)
    return rankings

  def _gibbs_sampler_pl_iteration(
      self,
      key: jnp.ndarray,
      lam: jnp.ndarray,
      rankings: jnp.ndarray,
      selectors: List[List[pl_exhaustive.Selector]],
      shape_lam: jnp.ndarray,
      rate_lam: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Innermost loop for the Gibbs sampler.

    This function completes an iteration for the Gibbs sampler that involves
    sampling arrival times (tau), plausibilities (lambda), and full rankings
    (sigma) from their full conditionals, based on the samples from the previous
    iteration.

    Args:
      key: PRNG key.
      lam: num_classes sized array of plausibilities.
      rankings: num_readers x num_classes sized array of rankings
      selectors: A nested list of selectors where each selector stands for a
        group of tied options for a reader, i.e. B_{r, m}
      shape_lam: num_classes sized array of shape hyperparameters for lambda
      rate_lam: num_classes sized array of rate hyperparameters for lambda

    Returns:
      key: PRNG key for the next iteration
      lam: num_classes sized array of plausibilities
      rankings: num_readers x num_classes sized array of rankings
    """
    num_readers, _ = rankings.shape
    keys = jax.random.split(key, num_readers + 3)
    tau = self._vmapped_sample_tau_given_lam_and_rankings(
        keys[:-3], lam, rankings
    )
    lam = self._sample_lam_given_tau(keys[-3], tau, shape_lam, rate_lam)
    rankings = self._sample_perm_given_lam_and_partial_rankings(
        keys[-2], rankings, selectors, lam
    )
    key = keys[-1]
    return key, lam, rankings

  def _initialize_rankings(
      self,
      key: jnp.ndarray,
      selectors: List[List[pl_exhaustive.Selector]],
      num_classes: int,
  ) -> jnp.ndarray:
    """Given a set of partial rankings, creates complying random full rankings.

    Given a set of partial rankings this function creates complying random full
    rankings for each partial ranking. This is a helper function that is tasked
    to create this representation which is useful downstream, so the actual
    ordering is not important as long as the partial rankings are obeyed.

    Args:
      key: PRNG key
      selectors: A nested list of selectors where each selector stands for a
        group of tied options for a reader, i.e. B_{r, m}
      num_classes: Total number of options / classes to be ranked

    Returns:
      rankings: num_readers x num_classes sized array of rankings, containing
        placeholder values that obey the partial orderings
    """
    num_readers = len(selectors)
    rankings = jnp.zeros((num_readers, num_classes), dtype=int)
    for reader, reader_selectors in enumerate(selectors):
      fake_lam = jnp.ones(num_classes)
      idx = 0
      for selector in reader_selectors:
        rankings = rankings.at[reader, idx : idx + len(selector)].set(
            jnp.array(selector)
        )
        fake_lam = fake_lam.at[jnp.array(selector)].set(0)
        idx = idx + len(selector)
      lkey, key = jax.random.split(key)
      probs = fake_lam / jnp.sum(fake_lam)
      last_ranks = jax.random.choice(
          key=lkey,
          a=jnp.arange(num_classes),
          shape=((num_classes,)),
          replace=False,
          p=probs,
      )
      rankings = rankings.at[reader, idx:].set(last_ranks[:-idx])
    return rankings
