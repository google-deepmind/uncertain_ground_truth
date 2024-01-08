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

"""Methods to aggregate partial rankings."""

import jax
import jax.numpy as jnp


def aggregate_irn(rankings: jnp.ndarray, groups: jnp.ndarray) -> jnp.ndarray:
  """Aggregate (partial) reader rankings using IRN.

  Args:
    rankings: Ranking of all classes as `num_examples x num_readers x
      num_classes` array.
    groups: Groups of all classes as `num_examples x num_readers x num_classes`
      array; -1 indicate unranked classes.

  Returns:
    Aggregated probabilities.
  """
  num_examples, num_readers, num_classes = rankings.shape
  indices = jnp.argsort(rankings, axis=2)
  static_indices = jnp.indices((num_examples, num_readers, num_classes))
  ordered_groups = groups[static_indices[0], static_indices[1], indices]
  mask = (ordered_groups >= 0)
  # We need the mask in the denominator to make sure we do not get NaN
  # when we have group -1.
  probabilities = jnp.sum(mask * 1. / (1 + mask * ordered_groups), axis=1)
  probabilities = probabilities / jnp.expand_dims(
      jnp.sum(probabilities, axis=1), axis=1)
  return probabilities


def sample_prirn(
    rng: jnp.ndarray,
    plausibilities: jnp.ndarray,
    num_samples: int,
    temperature: float,
    alpha: float,
) -> jnp.ndarray:
  """Sample plausibilities from IRN using Dirichlet.

  Args:
    rng: Random key.
    plausibilities: IRN plausibilities of shape `num_examples x num_classes`.
    num_samples: Number of plausibilities to sample.
    temperature: Temperature to apply to the given IRN plausibilities before
      sampling.
    alpha: Bias to apply to the given IRN plausibilities before sampling.

  Returns:
    Sampled plausibilities of shape `num_examples x num_samples x num_classes`.
  """
  num_examples, _ = plausibilities.shape
  return jax.random.dirichlet(
      rng,
      plausibilities * temperature + alpha,
      shape=(num_samples, num_examples),
  ).transpose((1, 0, 2))
