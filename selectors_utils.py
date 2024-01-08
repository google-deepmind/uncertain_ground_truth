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

"""Utilities for selectors data format."""

import enum
import functools
from typing import Any, Union

import jax
import jax.numpy as jnp
import numpy as np


class SelectorsStatus(enum.Enum):
  """Enum for errors in `check_selectors`."""
  NO_ERROR = 0
  INVALID_TYPE = 1
  EMPTY_READER_OR_GROUP = 2
  DUPLICATES_IN_READER = 3

  def is_valid(self):
    """Allows to easily check for NO_ERROR.

    Mainly used for `check_selectors` such that

        if not check_selectors(selectors, num_classes).is_valid():
          # Handle error.

    can be used to handle errors in selectors.

    Returns:
      Error as integer value
    """
    return self == SelectorsStatus.NO_ERROR


def _leaf_is_int(x: Any) -> bool:
  """Helper for jax.tree_util to check if a leaf in selectors is an integer.

  Args:
    x: value of leaf

  Returns:
    True if value is an integer
  """
  valid_types = [int, np.int64, np.int32, jnp.int32, jnp.int64]
  return any(isinstance(x, valid_type) for valid_type in valid_types)


def _block_is_leaf(x: Any) -> bool:
  """Helper to redefine leaves to be the groups in the selectors.

  Args:
    x: leaf value or list

  Returns:
    True if x corresponds to a block of labels
  """
  if not isinstance(x, list):
    return False
  return all(_leaf_is_int(y) for y in x)


def _reader_is_leaf(x: Any):
  """Helper to redefine leaves to be readers.

  Args:
    x: leaf value or list

  Returns:
    True if x corresponds to a reader of blocks of labels
  """
  if not isinstance(x, list):
    return False
  return all(_block_is_leaf(y) for y in x)


def _example_is_leaf(x: Any):
  """Helper to redefine leaves to be examples.

  Args:
    x: leaf value or list

  Returns:
    True if x corresponds to an example of readers of blocks of labels
  """
  if not isinstance(x, list):
    return False
  return all(_reader_is_leaf(y) for y in x)


def check_selectors(selectors: list[list[list[list[int]]]],
                    num_classes: int) -> SelectorsStatus:
  """Checks the integrity of selectors.

  Checks for (a) empty readers, (b) empty groups and (c)
  for integers as elements (even though this can be captured by typing).

  Can be applied on both individual examples (consisting of multiple readers)
  or on multiple examples.

  Args:
    selectors: reader rankings in selector representation to check
    num_classes: number of classes corresponding to selectors

  Returns:
    SelectorsStatus indicating the error found
  """
  # Will ignore all empty readers or blocks
  flattened_selectors, _ = jax.tree_util.tree_flatten(selectors)

  def is_label_valid(label: int) -> bool:
    """Helper to check that the label is valid."""
    return not _leaf_is_int(label) or label >= num_classes or label < 0

  if any(is_label_valid(label) for label in flattened_selectors):
    return SelectorsStatus.INVALID_TYPE

  # This will reduce all leaves = blocks/readers to either True/False depending
  # on whether the blocks/readers are empty.
  empty_readers_or_blocks = jax.tree_util.tree_map(
      lambda x: isinstance(x, list) and not x,
      selectors,
      is_leaf=_block_is_leaf)
  tree_any = functools.partial(jax.tree_util.tree_reduce, lambda x, y: x or y)
  if tree_any(empty_readers_or_blocks):
    return SelectorsStatus.EMPTY_READER_OR_GROUP

  def has_duplicates(x):
    """Check for duplicate labels in one reader ranking."""
    labels, _ = jax.tree_util.tree_flatten(x)
    return len(list(set(labels))) != len(labels)

  # This will reduce all readers to either True/False depending
  # on whether there are duplicates.
  duplicate_labels_in_readers = jax.tree_util.tree_map(
      has_duplicates, selectors, is_leaf=_reader_is_leaf)
  if tree_any(duplicate_labels_in_readers):
    return SelectorsStatus.DUPLICATES_IN_READER

  # All labels are correct and there are no empty readers or blocks.
  return SelectorsStatus.NO_ERROR


def limit_selectors_group_size(selectors: list[list[list[list[int]]]],
                               max_group_size: int) -> jnp.ndarray:
  """Limit the group sizes for a given dataset.

  This function breaks down groups of larger size into smaller ones iteratively.
  For an example grouping of [[0, 1, 2, 3], [4, 5, 6]] and a maximum group size
  of 3, the new grouping becomes [[0, 1, 2], [3, 4, 5], [6]].

  Args:
    selectors: Nested list of integers with lowermost lists corresponding to
      selectors for a specific instance, reader, and group.
    max_group_size: the maximum size any group can have, has to be positive.

  Returns:
    New selectors with the maximum number of groups reduced.
  """

  def _limit_group_size_in_reader(
      selectors: list[list[int]]) -> list[list[int]]:
    """Limits the size of groups within a single reader's groups.

    Args:
      selectors: Reader selectors that will be limited in size.

    Returns:
      Limited size selectors for the reader.
    """
    i = 0
    while i < len(selectors):
      if len(selectors[i]) > max_group_size:
        selectors.insert(i + 1, selectors[i][max_group_size:])
        selectors[i] = selectors[i][:max_group_size]
      i += 1
    return selectors

  return jax.tree_util.tree_map(
      _limit_group_size_in_reader, selectors, is_leaf=_reader_is_leaf)


def pad_selectors(
    selectors: list[list[list[list[int]]]], max_readers: int
) -> list[list[list[list[int]]]]:
  """Pad empty readers to have the same number of readers for each example.

  Args:
    selectors: Reader selectors that will be padded to `max_readers`.
    max_readers: Total number of readers to have for each example.

  Returns:
    New selectors with padded empty readers.
  """

  def _pad_readers(readers: list[list[list[int]]]) -> list[list[list[int]]]:
    """Pads readers for a single example.

    Args:
      readers: Reader selectors to pad.

    Returns:
      Padded readers for example.
    """
    if len(readers) < max_readers:
      return readers + [[]] * int(max_readers - len(readers))
    else:
      return readers[:max_readers]

  return jax.tree_util.tree_map(
      _pad_readers, selectors, is_leaf=_example_is_leaf)


def _array_is_leaf(x: Any) -> bool:
  """Helper for jax.tree_util to check if a leaf in selectors is an integer.

  Args:
    x: value of leaf

  Returns:
    True if value is an array
  """
  return isinstance(x, np.ndarray) or isinstance(x, jnp.ndarray)


def pad_rankings(
    rankings: list[list[Union[np.ndarray, jnp.ndarray, list[int]]]],
    groups: list[list[Union[np.ndarray, jnp.ndarray, list[int]]]],
    max_readers: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Pads rankings and groups.

  Pads rankings and groups obtained from the `data.partial_rankings_df'
  dataframe of `data.derm_assist.get_derm_dataframes` to have `max_readers`
  rankings and groups per example.

  Example usage:

    rankings = data.partial_rankings_df.groupby(
        'case_id')['ranking'].apply(list).tolist()
    groups = data.partial_rankings_df.groupby(
        'case_id')['groups'].apply(list).tolist()
    rankings, groups = selectors_utils.pad_rankings(
    rankings, groups, max_readers)

  Args:
    rankings: Ranking of all classes as `num_examples`-length list containing
      lists of array or list[int]; the number of readers per example is variable
      but the number of classes is fixed to `num_classes`.
    groups: Groups of all classes as `num_examples`-length list containing lists
      of array or list[int]; as above the number of readers per example is
      variable.
    max_readers: The number of maximum readers to pad to.

  Returns:
    Rankings and groups as `num_examples x max_readers x num_classes` shaped
    array each.
  """

  def _convert_to_list(x: Any):
    """Convert array to list of integers.

    Args:
      x: Input array.

    Returns:
      List containing the same values converted to integers.
    """
    # Need to check to handle the list[list[list[int]]] case properly.
    if _array_is_leaf(x):
      return list(x.astype(int))
    return x

  # Covert everything to list[list[list[int]]]:
  rankings = jax.tree_util.tree_map(
      _convert_to_list, rankings, is_leaf=_array_is_leaf
  )
  groups = jax.tree_util.tree_map(
      _convert_to_list, groups, is_leaf=_array_is_leaf
  )

  def _pad_rankings(list_of_rankings: list[list[int]]) -> list[list[int]]:
    """Padding rankings.

    Args:
      list_of_rankings: List of rankings, each encoded as list of integers.

    Returns:
      Padded list of rankings by adding dummy rankings to obtain
      `max_readers` rankings in total.
    """
    num_classes = len(list_of_rankings[0])
    num_padded = max_readers - len(list_of_rankings)
    empty_ranking = list(range(num_classes))
    return list_of_rankings + [empty_ranking] * num_padded

  def _pad_groups(list_of_groups: list[list[int]]) -> list[list[int]]:
    """Padding groups.

    Args:
      list_of_groups: List of groups, each encoded as list of integers.

    Returns:
      Padded list of groups by adding dummy groups filled with -1 to obtain
      `max_readers` groups in total.
    """
    # There are no cases with no readers:
    num_classes = len(list_of_groups[0])
    num_padded = max_readers - len(list_of_groups)
    empty_group = [-1] * num_classes
    return list_of_groups + [empty_group] * num_padded

  rankings_num_readers = jnp.array(
      [len(list_of_rankings) for list_of_rankings in rankings]
  )
  groups_num_readers = jnp.array(
      [len(list_of_groups) for list_of_groups in groups]
  )
  if jnp.any(rankings_num_readers != groups_num_readers):
    raise ValueError('Number of readers in rankings and groups does not match.')
  if jnp.any(rankings_num_readers > max_readers):
    raise ValueError('An example has more than max_readers readers.')

  # Renaming the leaf function for readability.
  is_example = _reader_is_leaf
  rankings = jnp.array(
      jax.tree_util.tree_map(_pad_rankings, rankings, is_leaf=is_example)
  )
  groups = jnp.array(
      jax.tree_util.tree_map(_pad_groups, groups, is_leaf=is_example)
  )
  return rankings, groups


def repeat_selectors(
    selectors: list[list[list[list[int]]]], num_repetitions: int
) -> list[list[list[list[int]]]]:
  """Repeat readers in selectors.

  Args:
    selectors: Input rankings as selectors.
    num_repetitions: Number of repetitions for each reader.

  Returns:
    Selectors with every reader repeated `num_repetitions` times.
  """

  def _repeat_readers(readers: list[list[list[int]]]) -> list[list[list[int]]]:
    """Repeats readers for a single example."""
    repeated_readers = []
    for reader in readers:
      repeated_readers += [reader] * num_repetitions
    return repeated_readers

  return jax.tree_util.tree_map(
      _repeat_readers, selectors, is_leaf=_example_is_leaf
  )
