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

"""Tests for IRN aggregation."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

import irn


class IrnTest(parameterized.TestCase):

  def test_sample_prirn(self):
    plausibilities = jnp.array([
        [0.4, 0.3, 0.2, 0.1],
        [0.8, 0.1, 0.05, 0.05],
    ])
    num_examples, num_classes = plausibilities.shape
    num_samples = 10
    sample_prirn = jax.jit(
        functools.partial(
            irn.sample_prirn,
            num_samples=num_samples,
            temperature=1e6,
            alpha=0,
        )
    )
    sampled_plausibilities = sample_prirn(jax.random.PRNGKey(0), plausibilities)
    np.testing.assert_array_equal(
        sampled_plausibilities.shape, (num_examples, num_samples, num_classes)
    )
    for i in range(num_samples):
      np.testing.assert_array_almost_equal(
          plausibilities,
          sampled_plausibilities[:, i],
          decimal=2,
      )


if __name__ == '__main__':
  absltest.main()
