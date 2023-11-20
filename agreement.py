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

"""Reader agreement evaluation."""

from typing import Callable

import jax.numpy as jnp
import numpy as np

import classification_metrics
import formats
import irn as aggregation


AgreementMetric = Callable[
    [jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    jnp.ndarray,
]


def topk(
    scores: np.ndarray,
    k: int,
    include_ties_for_kth: bool = True,
    remove_zero_entries: bool = True,
) -> np.ndarray:
  """Finds which scores are in the top-k scores.

  Args:
    scores: Array of shape (num_examples, num_classes) containing class scores.
    k: The number of classes to select. Must be positive.
    include_ties_for_kth: Whether to include ties for the k'th position. If
      False, then each row of the returned array will contain exactly k Trues
      and first of the tying scores will be taken. It true, then rows with
      scores tying for the kth position will have more than k True values.
      Defaults to True.
    remove_zero_entries: If sum(scores > 0) < k, avoids selecting zero entries
      in topk selector.

  Returns:
    Numpy array of shape (num_examples, num_classes) containing booleans
      where the prediction is one of the top-k predictions for that example.
      Each row will have exactly k Trues if there are no ties.
  Raises:
    ValueError if k is non-positive.
  """
  if k <= 0:
    raise ValueError(f'Top-k must be positive but got {k}.')
  k = min(k, scores.shape[1])
  thresholds = np.partition(scores, -k, axis=1)[:, -k, np.newaxis]
  result = scores >= thresholds
  if remove_zero_entries:
    result = np.logical_and(result, scores > 0)
  if include_ties_for_kth:
    return result
  else:
    # Fix rows with more than k selected due to ties by turning off classes
    # which ties for k'th from right to left.
    where_fix = result.sum(axis=1) > k
    where_tie = scores[where_fix] == thresholds[where_fix]
    num_extra = result[where_fix].sum(axis=1) - k
    fixed = result[where_fix]
    for i in range(fixed.shape[0]):
      for j in reversed(range(fixed.shape[1])):
        if where_tie[i, j]:
          fixed[i, j] = False
          num_extra[i] -= 1
          if num_extra[i] == 0:
            break
    result[where_fix] = fixed
    return result


def leave_one_reader_out_agreement(
    rankings: jnp.ndarray,
    groups: jnp.ndarray,
    num_readers: jnp.ndarray,
    agreement_metric: AgreementMetric,
) -> jnp.ndarray:
  """Compute agreement over readers using given metric function.

  Computes reader agreement in a leave-one-reader-out fashion using a given
  metric which could compute accuracy or average overlap given the
  reader rankings.

  Args:
    rankings: Rankings for all examples and readers as `num_examples x
      max_readers x num_classes` array.
    groups: Groups for all examples and readers `num_examples x max_readers x
      num_classes` array; -1 indicates unranked classes.
    num_readers: `num_examples` shaped array containing the number of readers
      per example.
    agreement_metric: Metric to apply between the left out and all other
      readers; it expects as inputs the rankings and groups for the left out
      reader and the remaining readers in addition to the number of readers per
      examples.

  Returns:
    Computed agreements as `num_examples x max_readers` array.
  """
  num_examples, max_readers, _ = rankings.shape
  agreements = jnp.zeros((num_examples, max_readers))
  for r in range(max_readers):
    left_out_mask = jnp.arange(max_readers) == r
    left_out_rankings = rankings[:, left_out_mask]
    left_out_groups = groups[:, left_out_mask]
    other_rankings = rankings[:, jnp.logical_not(left_out_mask)]
    other_groups = groups[:, jnp.logical_not(left_out_mask)]
    agreement = agreement_metric(
        left_out_rankings,
        left_out_groups,
        other_rankings,
        other_groups,
        num_readers,
    )
    agreements = agreements.at[:, r].set(agreement)
  return agreements


def leave_one_reader_out_coverage_agreement(
    rankings: jnp.ndarray, groups: jnp.ndarray, num_readers: jnp.ndarray
) -> jnp.ndarray:
  """Computes reader agreement using standard coverage.

  Computes agreement using standard coverage against the top-1 IRN label by
  taking the ranking of reach left out reader as variable-sized prediction
  set.

  Args:
    rankings: Rankings for all examples and readers as `num_examples x
      max_readers x num_classes` array.
    groups: Groups for all examples and readers `num_examples x max_readers x
      num_classes` array; -1 indicates unranked classes.
    num_readers: `num_examples` shaped array containing the number of readers
      per example.

  Returns:
    Computed standard coverage agreements as `num_examples x max_readers` array.
  """

  def standard_coverage_agreement(
      left_out_rankings: jnp.ndarray,
      left_out_groups: jnp.ndarray,
      other_rankings: jnp.ndarray,
      other_groups: jnp.ndarray,
      num_readers: jnp.ndarray,  # pylint: disable=unused-argument
  ) -> jnp.ndarray:
    """Agreement metric to compute standard coverage against top-1 IRN.

    Args:
      left_out_rankings: Rankings of the left-out reader as `num_examples x 1 x
        num_classes` array.
      left_out_groups: Groups of the left-out reader as `num_examples x 1 x
        num_classes` array.
      other_rankings: Rankings of the remaining readers as `num_examples x
        num_other_readers x num_classes` array.
      other_groups: Groups of the remaining readers as `num_examples x
        num_other_readers x num_classes` array.
      num_readers: Number of readers for each example as `num_examples` array.

    Returns:
        Standard coverage of left out reader converted to prediction set
        evaluated against the top-1 IRN label from the other readers.
    """
    prediction_sets = formats.convert_rankings_to_prediction_sets(
        jnp.squeeze(left_out_rankings), jnp.squeeze(left_out_groups)
    )
    plausibilities = aggregation.aggregate_irn(other_rankings, other_groups)
    labels = jnp.array(topk(np.array(plausibilities), 1)).astype(int)
    return classification_metrics.aggregated_coverage(prediction_sets, labels)

  return leave_one_reader_out_agreement(
      rankings, groups, num_readers, standard_coverage_agreement
  )
