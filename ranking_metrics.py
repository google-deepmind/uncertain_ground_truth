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

"""Metrics to compare a pair of rankings or partial rankings.

Average overlap:
The metric is calculated by averaging the size of the overlaps between the
first k elements of predicted rankings and rankings divided by k,
(size of the overlap at k) / k, where k goes from 1 to maximum length,
specified by `max_lengths`. If maximum length is 0, we return 0.
"""

import jax
import jax.numpy as jnp


def average_overlap(
    predicted_rankings: jnp.ndarray,
    rankings: jnp.ndarray,
    max_lengths: jnp.ndarray,
) -> jnp.ndarray:
  """Calculates the average overlap metric.

  AO(r*, r) = trace[K_n * T_n * P(r*) * (T_n * P(r))^T]

  r* - Predicted ranking (`predicted_rankings`),
  r - Ranking (`rankings`),
  n - Number of classes,
  m - Maximum overlap length to consider where m < n (`max_lengths`),
  K_n - Diagonal coefficient matrix with entries
    [1/n, 1/2n, 1/3n, ... 1/mn, 0, 0 ... 0],
  T_n - Lower triangle of a matrix of ones with size n times n,
  P(.) - Ranking (permutation) matrix corresponding to the given ranking.

  The `predicted_rankings` and `rankings` are expected to give an ordering of
  all the classes 0, 1, 2, ... n as integers in which each class occurs exactly
  once.

  Args:
    predicted_rankings: The predicted ranking with shape (num_examples,
      num_classes).
    rankings: Ranking with shape (num_examples, num_classes).
    max_lengths: The maximum lengths of the overlap set to consider with shape
      (num_examples,).

  Returns:
    average_overlap: The average overlap metric with shape (num_examples,).
  """
  return jax.vmap(average_overlap_single)(predicted_rankings, rankings,
                                          max_lengths)


def average_overlap_single(
    predicted_ranking: jnp.ndarray,
    ranking: jnp.ndarray,
    max_length: jnp.ndarray,
) -> jnp.ndarray:
  """Calculates the AO metric for a single instance.

  Args:
    predicted_ranking: The predicted ranking with shape (num_classes,).
    ranking: Ranking with shape (num_classes,).
    max_length: The maximum length of the overlap set to consider.

  Returns:
    average_overlap: The average overlap metric.
  """
  num_classes = len(ranking)

  coeff_matrix_diagonal = ((jnp.arange(num_classes) < max_length) /
                           jnp.arange(1, num_classes + 1))
  coeff_matrix = jnp.diag(coeff_matrix_diagonal)

  tril = jnp.tril(jnp.ones((num_classes, num_classes)))
  ranking_matrix = jax.nn.one_hot(ranking, num_classes)
  predicted_ranking_matrix = jax.nn.one_hot(predicted_ranking, num_classes)

  ao_sum = jnp.trace(coeff_matrix @ tril @ predicted_ranking_matrix
                     @ (tril @ ranking_matrix).T)

  return jnp.where(max_length > 0, ao_sum / max_length, 0.0)


def partial_average_overlap(
    predicted_rankings: jnp.ndarray,
    predicted_groups: jnp.ndarray,
    rankings: jnp.ndarray,
    groups: jnp.ndarray,
    max_lengths: jnp.ndarray,
    normalize: bool = True,
) -> jnp.ndarray:
  """Calculates the partial average overlap metric.

  UAO(r*, r) = trace[K_n * T_n * P(r*) * (T_n * P(r))^T] - unnormalized
  AO(r*, r) = UAO(r*, r) / sqrt(UAO(r*, r*) x UAO(r, r)) - normalized

  r* - Predicted partial ranking (`predicted_rankings` + `predicted_groups`),
  r - Partial ranking (`rankings` + `groups`),
  n - Number of classes,
  m - Maximum overlap length to consider where m < n (`max_lengths`),
  K_n - Diagonal coefficient matrix with entries
    [1/n, 1/2n, 1/3n, ... 1/mn, 0, 0 ... 0],
  T_n - Lower triangle of a matrix of ones with size n times n,
  P(.) - Partial ranking (permutation) matrix corresponding to the given partial
    ranking.

  The partial average overlap of a partial ranking with itself is not
  necessarily 1. This makes the metric difficult to interpret, which is why we
  apply normalization as shown above (when `normalize` is set to True), which is
  similar to the normalization of the correlation coefficient.

  The calculation of partial average overlap differs from average overlap in the
  calculation of the ranking matrix. The partial ranking matrix is a permutation
  matrix to express ranking with ties. The entry {i, j} in the matrix represents
  the weight of class j being in place i in the ranking.

  The `predicted_rankings` and `rankings` are expected to give an ordering of
  all the classes 0, 1, 2, ... n as integers in which each class occurs exactly
  once.

  The groups provide information about ties in the ranking. For example if we
  have a ranking [0, 1, 2, 3, 4] and groups [0, 1, 1, 1, -1] indicates that
  there is a tie between classes 1, 2 and 3. -1 indicates an unranked group
  which must be last.

  Note that if either predicted ranking or ranking is empty (groups contain only
  -1) the method returns 0.

  Args:
    predicted_rankings: The predicted ranking with shape (num_examples,
      num_classes).
    predicted_groups: The predicted ranking groups with shape (num_examples,
      num_classes).
    rankings: Ranking with shape (num_examples, num_classes).
    groups: The ranking groups with shape (num_examples, num_classes).
    max_lengths: The maximum lengths of the overlap set to consider with shape
      (num_examples,).
    normalize: Whether to normalize the metric.

  Returns:
    partial_average_overlap: The partial average overlap metric with shape
      (num_examples,).
  """
  return jax.vmap(
      partial_average_overlap_single,
      in_axes=(0, 0, 0, 0, 0, None),
  )(
      predicted_rankings,
      predicted_groups,
      rankings,
      groups,
      max_lengths,
      normalize,
  )


def partial_average_overlap_single(
    predicted_ranking: jnp.ndarray,
    predicted_groups: jnp.ndarray,
    ranking: jnp.ndarray,
    groups: jnp.ndarray,
    max_length: jnp.ndarray,
    normalize: bool = True,
) -> jnp.ndarray:
  """Calculates the AO metric from partial rankings for a single instance.

  Args:
    predicted_ranking: The predicted ranking with shape (num_classes,).
    predicted_groups: The predicted grouping of classes with shape
      (num_classes,).
    ranking: Ranking with shape (num_classes,).
    groups: The grouping of classes with shape (num_classes,).
    max_length: The maximum length of the overlap set to consider.
    normalize: Whether to normalize the metric.

  Returns:
    partial_average_overlap: The average overlap metric.
  """
  unnormalized_partial_ao = _unnormalized_partial_ao(
      predicted_ranking,
      predicted_groups,
      ranking,
      groups,
      max_length,
  )

  if not normalize:
    return unnormalized_partial_ao

  predicted_self_ao = _unnormalized_partial_ao(
      predicted_ranking=predicted_ranking,
      predicted_groups=predicted_groups,
      ranking=predicted_ranking,
      groups=predicted_groups,
      max_length=max_length,
  )
  self_ao = _unnormalized_partial_ao(
      predicted_ranking=ranking,
      predicted_groups=groups,
      ranking=ranking,
      groups=groups,
      max_length=max_length,
  )
  return jnp.where(
      unnormalized_partial_ao > 0,
      unnormalized_partial_ao / jnp.sqrt(predicted_self_ao * self_ao),
      0.0,
  )


def _unnormalized_partial_ao(
    predicted_ranking: jnp.ndarray,
    predicted_groups: jnp.ndarray,
    ranking: jnp.ndarray,
    groups: jnp.ndarray,
    max_length: jnp.ndarray,
) -> jnp.ndarray:
  """Calculates unnormalized AO from partial rankings for a single instance.

  Args:
    predicted_ranking: The predicted ranking with shape (num_classes,).
    predicted_groups: The predicted grouping of classes with shape
      (num_classes,).
    ranking: Ranking with shape (num_classes,).
    groups: The grouping of classes with shape (num_classes,).
    max_length: The maximum length of the overlap set to consider.

  Returns:
    partial_average_overlap: The average overlap metric.
  """
  num_classes = len(ranking)

  coeff_matrix_diagonal = ((jnp.arange(num_classes) < max_length) /
                           jnp.arange(1, num_classes + 1))
  coeff_matrix = jnp.diag(coeff_matrix_diagonal)

  tril = jnp.tril(jnp.ones((num_classes, num_classes)))
  ranking_matrix = _build_partial_ranking_matrix(ranking, groups)
  predicted_ranking_matrix = _build_partial_ranking_matrix(
      predicted_ranking, predicted_groups)

  ao_sum = jnp.trace(coeff_matrix @ tril @ predicted_ranking_matrix
                     @ (tril @ ranking_matrix).T)

  # If max_length is 0 return 0.
  ao = jnp.where(max_length > 0, ao_sum / max_length, 0.0)

  # If predicted_ranking or ranking is empty (groups are all -1) return 0.
  def is_ranking_empty(groups: jnp.ndarray) -> jnp.ndarray:
    return jnp.all(groups == -1)

  ao = jnp.where(is_ranking_empty(predicted_groups), 0.0, ao)
  ao = jnp.where(is_ranking_empty(groups), 0.0, ao)

  return ao


def _build_partial_ranking_matrix(
    ranking: jnp.ndarray,
    groups: jnp.ndarray,
) -> jnp.ndarray:
  """Builds the partial ranking matrix.

  The partial ranking matrix is a permutation matrix to express ranking with
  ties. The entry {i, j} in the matrix represents the weight of class j
  being in place i in the ranking.

  Args:
    ranking: Ranking with shape (num_classes,).
    groups: The grouping of classes with shape (num_classes,).

  Returns:
    partial_ranking_matrix: The partial ranking matrix.
  """
  num_classes = len(ranking)

  def build_row(group: jnp.ndarray) -> jnp.ndarray:
    group_mask = groups == group
    group_one_hot = jax.nn.one_hot(
        jnp.where(group_mask, ranking, -1),
        num_classes=num_classes,
    )
    return jnp.sum(group_one_hot, axis=0) / jnp.sum(group_mask)

  return jax.vmap(build_row)(groups)
