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

"""Tests for gaussian_toy_dataset."""

from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np

import gaussian_toy_dataset as gtd


class GaussianToyDatasetTest(parameterized.TestCase):

  @parameterized.parameters([
      dict(
          class_weights=np.array([0.5, 0.5]),
          class_sigmas=0.1,
          dimensionality=10,
          sigma=1),
      dict(
          class_weights=np.array([0.5, 0.5]),
          class_sigmas=0.1,
          dimensionality=10,
          sigma=0.01),
      dict(
          class_weights=np.array([0.5, 0.5]),
          class_sigmas=0.1,
          dimensionality=1000,
          sigma=1),
      dict(
          class_weights=np.array([1, 1]),
          class_sigmas=0.1,
          dimensionality=10,
          sigma=1),
      dict(
          class_weights=np.array([0.1, 0.1]),
          class_sigmas=0.1,
          dimensionality=10,
          sigma=1),
      dict(
          class_weights=np.array([0.1, 0.1]),
          class_sigmas=np.array([0.1, 0.5]),
          dimensionality=10,
          sigma=1),
  ])
  def test_constructor(self, class_weights, class_sigmas, dimensionality,
                       sigma):
    model = gtd.GaussianToyDataset(
        gtd.PRNGSequence(0), class_weights, class_sigmas, dimensionality, sigma
    )
    self.assertIsNotNone(model.means)
    self.assertTrue(jnp.all(model.sigmas > 0))
    self.assertTrue(jnp.isclose(1, jnp.sum(model.class_probabilities)))

  @parameterized.parameters([
      dict(
          class_weights=np.array([]),
          class_sigmas=0.1,
          dimensionality=10,
          sigma=1),
      dict(
          class_weights=np.array([0.5, 0.5]),
          class_sigmas=np.array([1]),
          dimensionality=10,
          sigma=1),
  ])
  def test_constructor_errors(self, class_weights, class_sigmas, dimensionality,
                              sigma):
    with self.assertRaises(ValueError):
      gtd.GaussianToyDataset(
          gtd.PRNGSequence(0),
          class_weights,
          class_sigmas,
          dimensionality,
          sigma,
      )

  @parameterized.parameters([
      dict(
          class_weights=np.array([1]),
          class_sigmas=0.1,
          dimensionality=10,
          num_examples=100),
      dict(
          class_weights=np.array([1, 1]),
          class_sigmas=0.1,
          dimensionality=10,
          num_examples=100),
      dict(
          class_weights=np.array([1, 1, 1, 1]),
          class_sigmas=0.1,
          dimensionality=10,
          num_examples=100),
      dict(
          class_weights=np.array([1, 1, 1, 1]),
          class_sigmas=np.array([0.5, 0.5, 0.3, 0.1]),
          dimensionality=10,
          num_examples=100),
  ])
  def test_sample_points(self, class_weights, class_sigmas, dimensionality,
                         num_examples):
    model = gtd.GaussianToyDataset(
        gtd.PRNGSequence(0), class_weights, class_sigmas, dimensionality, 1
    )
    examples, ground_truth = model.sample_points(num_examples)
    self.assertEqual(examples.shape[0], num_examples)
    self.assertEqual(examples.shape[1], dimensionality)
    if class_weights.shape[0] == 1:
      self.assertTrue(jnp.all(ground_truth == 0))
    else:
      for k in range(class_weights.shape[0]):
        self.assertTrue(jnp.any(ground_truth == k))

  @parameterized.parameters([
      dict(num_examples=0),
      dict(num_examples=-1),
  ])
  def test_sample_points_errors(self, num_examples):
    with self.assertRaises(ValueError):
      model = gtd.GaussianToyDataset(
          gtd.PRNGSequence(0), jnp.array([1, 1]), 0.1, 10, 1
      )
      model.sample_points(num_examples)

  @parameterized.parameters([
      dict(class_weights=np.array([1]), dimensionality=10, num_examples=100),
      dict(class_weights=np.array([1, 1]), dimensionality=10, num_examples=100),
      dict(
          class_weights=np.array([1, 1, 1, 1]),
          dimensionality=10,
          num_examples=100),
  ])
  def test_evaluate_points(self, class_weights, dimensionality, num_examples):
    model = gtd.GaussianToyDataset(
        gtd.PRNGSequence(0), class_weights, 0.1, dimensionality, 1
    )
    points, _ = model.sample_points(num_examples)
    probabilities = model.evaluate_points(points)
    self.assertEqual(probabilities.shape[0], points.shape[0])
    self.assertEqual(probabilities.shape[1], class_weights.shape[0])
    self.assertTrue(jnp.all(jnp.isclose(jnp.sum(probabilities, axis=1), 1)))
    self.assertTrue(jnp.all(probabilities >= 0))

  def test_evaluate_points_priors(self):
    model = gtd.GaussianToyDataset(
        gtd.PRNGSequence(0), jnp.array([0.1, 0.9]), 1, 1
    )
    # Ensure that there is not much overlap to have probabilities be
    # determined primarily by the class priors.
    model.means = jnp.array([[-0.9], [0.9]])
    num_examples = 10000
    _, labels = model.sample_points(num_examples)
    self.assertLessEqual(jnp.sum(labels == 0), (0.1 + 1e-2) * num_examples)
    self.assertGreaterEqual(jnp.sum(labels == 1), (0.9 - 1e-2) * num_examples)

  @parameterized.parameters([
      dict(points=np.zeros((100))),
      dict(points=np.zeros((100, 5))),
  ])
  def test_evaluate_points_errors(self, points):
    with self.assertRaises(ValueError):
      model = gtd.GaussianToyDataset(
          gtd.PRNGSequence(0), jnp.array([1, 1]), 0.1, 10, 1
      )
      model.evaluate_points(jnp.array(points))

  @parameterized.parameters([
      dict(num_examples=10, readers=np.array([0, 0, 0]), expected_length=None),
  ])
  def test_sample_rankings(self, num_examples, readers, expected_length):
    model = gtd.GaussianToyDataset(
        gtd.PRNGSequence(0), jnp.array([1, 1]), 0.1, 10, 1
    )
    points, _ = model.sample_points(num_examples)
    probabilities = model.evaluate_points(points)
    rankings, groups = model.sample_rankings(probabilities, jnp.array(readers),
                                             expected_length)
    self.assertEqual(rankings.shape[0], points.shape[0])
    self.assertEqual(rankings.shape[1], readers.shape[0])
    self.assertEqual(rankings.shape[2], model.num_classes)
    self.assertTrue(jnp.all(rankings < model.num_classes))
    self.assertTrue(jnp.all(rankings >= 0))
    np.testing.assert_array_equal(rankings.shape, groups.shape)

  @parameterized.parameters([
      dict(
          probabilities=np.zeros((100)),
          readers=np.zeros((2)),
          expected_length=None,
          grouping_threshold=None),
      dict(
          probabilities=np.zeros((100, 1)),
          readers=np.array([]),
          expected_length=None,
          grouping_threshold=None),
      dict(
          probabilities=np.zeros((100, 3)),
          readers=np.array([]),
          expected_length=None,
          grouping_threshold=None),
      dict(
          probabilities=np.zeros((100, 2)),
          readers=np.array([]),
          expected_length=None,
          grouping_threshold=None),
      dict(
          probabilities=np.zeros((100, 2)),
          readers=np.zeros((2, 2)),
          expected_length=None,
          grouping_threshold=None),
      dict(
          probabilities=np.zeros((100, 2)),
          readers=np.zeros((2)),
          expected_length=0,
          grouping_threshold=None),
      dict(
          probabilities=np.zeros((100, 2)),
          readers=np.zeros((2)),
          expected_length=0,
          grouping_threshold=0),
  ])
  def test_sample_rankings_errors(self, probabilities, readers, expected_length,
                                  grouping_threshold):
    with self.assertRaises(ValueError):
      model = gtd.GaussianToyDataset(
          gtd.PRNGSequence(0), jnp.array([1, 1]), 0.1, 10, 1
      )
      model.sample_rankings(probabilities, readers, expected_length,
                            grouping_threshold)

  def test_sample_partial_rankings(self):
    num_classes = 5
    probabilities = jnp.array([[
        [0.4, 0.35, 0.15, 0.1, 0],
        [0.6, 0.1, 0.1, 0.1, 0.1],
        [0.525, 0.475, 0, 0, 0],
        [0.6, 0.3, 0.1, 0, 0],
        [0.4, 0.3, 0.2, 0.1, 0],
    ]])
    rankings = jnp.tile(
        jnp.expand_dims(jnp.arange(num_classes), axis=0), num_classes)
    expected_groups_001 = jnp.array([[
        [0, 1, 2, 3, 4],
        [0, 1, 1, 1, 1],
        [0, 1, 2, 2, 2],
        [0, 1, 2, 3, 3],
        [0, 1, 2, 3, 4],
    ]])
    expected_groups_01 = jnp.array([[
        [0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 1, 2, 2, 2],
        [0, 0, 0, 0, 0],
    ]])
    expected_groups_005 = jnp.array([[
        [0, 0, 1, 1, 2],
        [0, 1, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 1, 2, 3, 3],
        [0, 1, 2, 3, 4],
    ]])
    model = gtd.GaussianToyDataset(
        gtd.PRNGSequence(0), jnp.ones(num_classes), 0.1, 10, 1
    )
    groups = model._sample_partial_rankings(
        probabilities, rankings, expected_length=None, grouping_threshold=0.01)
    np.testing.assert_equal(np.array(groups), np.array(expected_groups_001))
    groups = model._sample_partial_rankings(
        probabilities, rankings, expected_length=None, grouping_threshold=0.1)
    np.testing.assert_equal(np.array(groups), np.array(expected_groups_01))
    groups = model._sample_partial_rankings(
        probabilities, rankings, expected_length=None, grouping_threshold=0.05)
    np.testing.assert_equal(np.array(groups), np.array(expected_groups_005))

  def test_vary_number_of_readers(self):
    num_classes = 5
    num_examples = 10
    groups = jnp.array([
        [0, 1, 2, 3, 4],
        [0, 1, 1, 1, 1],
        [0, 1, 2, 2, 2],
        [0, 1, 2, 3, 3],
        [0, 1, 2, 3, 4],
    ])
    groups = jnp.repeat(jnp.expand_dims(groups, axis=0), num_examples, axis=0)
    model = gtd.GaussianToyDataset(
        gtd.PRNGSequence(0), jnp.ones(num_classes), 0.1, 10, 1
    )
    groups, num_readers = model.vary_number_of_readers(groups, 2)
    for n in range(num_examples):
      actual_readers = jnp.sum(groups[n] >= 0, axis=1)
      actual_readers = jnp.sum(actual_readers > 0, axis=0)
      self.assertEqual(num_readers[n], actual_readers)


if __name__ == "__main__":
  absltest.main()
