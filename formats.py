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

"""Functionality to convert formats of partial rankings.

There are three data formats to represent partial rankings used in the
repository.

1. The opinions format is a list of arrays per example with shape (max_readers,
num_classes) where rows correspond to readers and columns to classes. The arrays
contain confidence values for each reader/class pair. Confidence values must be
non-negative integers. The highest confidence values are ranked first and zero
confidence value corresponds to unranked classes. Rows with all zero values
indicate empty readers.

2. The selectors format is composed of four level of lists. The inner-most list
contains integers corresponding to classes. The other list levels correspond to
data points, readers and groups of classes with the same ranking.

3. The rankings/groups format is composed of two arrays `rankings` and `groups`
both with shape (num_examples, max_readers, num_classes). The `rankings` array
represents one possible ordering of the classes according to the partial
ranking. The `groups` array indicates ties in the ranking and contains
incrementing integers starting from 0. Unranked classes are assigned -1.

See equivalent examples of these formats below where `num_examples` = 2,
`max_readers` = 2 and `num_classes` = 4.

opinions = [
    np.array([[5, 0, 3, 5], [1, 2, 0, 0]]),
    np.array([[0, 0, 0, 0], [2, 2, 3, 0]]),
    np.array([[0, 0, 0, 0], [0, 0, 0, 0]]),
]

selectors = [
    [[[0, 3], [2]], [[1], [0]]],
    [[[2], [0, 1]]],
    [],
]

rankings = np.array([
    [[0, 3, 2, 1], [1, 0, 2, 3]],
    [[0, 1, 2, 3], [2, 0, 1, 3]],
    [[0, 1, 2, 3], [0, 1, 2, 3]],
])

groups = np.array([
    [[ 0,  0,  1, -1], [ 0,  1, -1, -1]],
    [[-1, -1, -1, -1], [ 0,  1,  1, -1]],
    [[-1, -1, -1, -1], [-1, -1, -1, -1]],
])

For prediction sets, a 0-1 encoding is used, for example the following
for `num_examples = 2` and `num_classes = 4` where the first set includes
classes 1 and 3 and the second one includes classes 0, 2 and 3. Note that
we usually predict one prediction set per example and not per reader.
These are usually derived from some sort of (conformity) scores that often
coincide with the logit or softmax output of a classifier.

prediction_sets = np.array([
    [0, 1, 0, 1],
    [1, 0, 1, 1],
])
"""

import itertools

import jax.numpy as jnp


def convert_rankings_to_selectors(
    rankings: jnp.ndarray,
    groups: jnp.ndarray,
) -> list[list[list[list[int]]]]:
  """Converts partial rankings from the rankings/groups to the selectors format.

  Args:
    rankings: One possible ranking of all classes corresponding to the partial
      ranking with shape (num_examples, max_readers, num_classes).
    groups: Groups array indicating the ties in the ranking with shape
      (num_examples, max_readers,  num_classes); -1 indicate unranked classes.

  Returns:
    selectors: The selectors format composed of four level of lists. The list
      levels correspond to data points, readers and groups of classes.
  """

  def get_splits(rankings, group_diffs):
    """Splits the ranking according to the grouping information."""
    return jnp.split(rankings, jnp.where(group_diffs)[0] + 1)

  group_diffs = groups[:, :, :-1] - groups[:, :, 1:]
  selectors = []
  num_examples, max_readers, _ = groups.shape
  for i in range(num_examples):
    reader_selectors = []
    for j in range(max_readers):
      # Depending on where the groups change, the splits are created.
      splits = get_splits(rankings[i, j], group_diffs[i, j])
      # Check if the last group is indeed the group of unranked classes and
      # exclude it if it is.
      if groups[i, j, -1] < 0:
        splits = splits[:-1]
      # The last group is the unranked one therefore is not indexed.
      reader_selector = [split.tolist() for split in splits]
      # If the reader did not rank anything, we do not want to include
      # anything. This is useful when converting groups where all classes
      # are assigned -1, i.e., unranked.
      if reader_selector:
        reader_selectors.append(reader_selector)
    selectors.append(reader_selectors)
  return selectors


def convert_selectors_to_rankings(
    selectors: list[list[list[list[int]]]],
    num_classes: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Converts partial rankings from the selectors to the rankings/groups format.

  Note that selectors must not contain any empty data points and the same
  number of readers per example is expected. This can be accomplished using
  `selectors_utils.pad_selectors`.

  Args:
    selectors: The selectors format composed of four level of lists. The list
      levels correspond to data points, readers and groups of classes.
    num_classes: The number of classes.

  Returns:
    rankings: One possible ranking of all classes corresponding to the partial
      ranking with shape (num_examples, max_readers, num_classes).
    groups: Groups array indicating the ties in the ranking with shape
      (num_examples, max_readers,  num_classes); -1 indicate unranked classes.
  """
  rankings = []
  groups = []
  for case in selectors:
    case_rankings = []
    case_groups = []
    for selector in case:
      ranking = list(itertools.chain.from_iterable(selector))
      group = []
      b = 0
      for block in selector:
        group += [b] * len(block)
        b += 1
      assert len(groups) == len(rankings)
      group += [-1] * (num_classes - len(group))
      ranking += list(set(range(num_classes)) - set(ranking))
      case_rankings.append(ranking)
      case_groups.append(group)
    rankings.append(case_rankings)
    groups.append(case_groups)
  return jnp.array(rankings), jnp.array(groups)


def convert_prediction_sets_to_rankings(
    conformity_scores: jnp.ndarray, prediction_sets: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Converts prediction sets from scores to rankings in rankings/groups format.

  We assume that the provided conformity scores do not have any ties.

  Args:
    conformity_scores: Conformity scores (p-values or logits also work) to sort
      classes by, shaped `num_examples x num_classes`.
    prediction_sets: Prediction sets to select included classes, shaped
      `num_examples x num_classes`.

  Returns:
    rankings: One possible ranking of all classes corresponding to the partial
      ranking with shape (num_examples, num_classes).
    groups: Groups array indicating the ties in the ranking with shape
      (num_examples, num_classes); -1 indicate unranked classes.
  """
  num_examples, num_classes = prediction_sets.shape
  # Sort in descending order to obtain ranking:
  rankings = jnp.flip(jnp.argsort(conformity_scores, axis=1), axis=1)
  groups = jnp.repeat(
      jnp.expand_dims(jnp.arange(num_classes), axis=0), num_examples, axis=0
  )
  mask = jnp.take_along_axis(prediction_sets, rankings, axis=1)
  groups = jnp.where(mask, groups, -1)
  return rankings, groups


def convert_rankings_to_prediction_sets(
    rankings: jnp.ndarray, groups: jnp.ndarray
) -> jnp.ndarray:
  """Converts partial rankings in rankings/groups format to prediction sets.

  Note that ties in the ranking are not relevant for converting to
  prediction sets, the prediction sets simply includes all ranked classes.

  Args:
    rankings: One possible ranking of all classes corresponding to the partial
      ranking with shape (num_examples, num_classes).
    groups: Groups array indicating the ties in the ranking with shape
      (num_examples, num_classes); -1 indicate unranked classes.

  Returns:
    predictions_sets: Prediction sets as 0-1 array of shape
      `num_examples x num_classes`.
  """
  indices = jnp.argsort(rankings, axis=1)
  ordered_groups = jnp.take_along_axis(groups, indices, 1)
  prediction_sets = (ordered_groups >= 0).astype(int)
  return prediction_sets
