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

"""Colab and plotting uilities."""

import matplotlib
from matplotlib import pyplot as plt
import numpy as np


# Pyplot colors minus green and red.
COLORS = [
    '#1f77b4',
    '#ff7f0e',
    # '#2ca02c',
    # '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
]
COLOR_RED = '#d62728'


def set_style():
  """Set plotting style."""
  font = {
      'family': 'sans-serif',
      'weight': 'normal',
      'size': 12,
  }
  # Set default color cycle for plots.
  matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=COLORS)
  matplotlib.rc('font', **font)
  plt.rcParams['figure.dpi'] = 100
  plt.rcParams['text.usetex'] = False


def plot_hist(  # pylint: disable=redefined-builtin
    values, bins=20, range=None, normalize=False, **kwargs
):
  """General utility to simplify histogram plotting with matplotlib.

  Args:
    values: One-dimensional array for which to plot a histogram.
    bins: Number of bins to plot; for discrete valuzes in values, this could be
      `np.max(values) - np.min(values) + 1`, while for continuous values this is
      variable and the range can be controlled by `range`.
    range: Range on the x-axis to plot, for example `range = (np.min(values),
      np.max(values))`.
    normalize: Whether to normalize the histogram values.
    **kwargs: Keyword arguments are passed to `plt.bar`.

  Returns:
    The histogram and bins as returned by `np.histogram`.
  """
  hist, bins = np.histogram(values, bins=bins, range=range)
  if normalize:
    hist = hist / np.sum(hist)
  width = 0.95 * (bins[1] - bins[0])
  center = (bins[:-1] + bins[1:]) / 2
  plt.bar(center, hist, align='center', width=width, **kwargs)
  return hist, bins


def plot_simplex(points, labels):
  """Plot 2D points with given labels in a simplex.

  Together with `project_simplex`, this allows to plot plausibilities
  with three classes in the 3-simplex of probability distributions using
  matplotlib's `tricontouf`.

  Args:
    points: `num_examples x 2` shaped array of points to plot; from
      `project_simplex`.
    labels: `num_examples` shaped arrays of labels to indicate which of the
      points are in the set to plot.
  """
  ax = plt.gca()
  ax.add_line(
      matplotlib.lines.Line2D(
          [0, 0.5, 1.0, 0], [0, np.sqrt(3) / 2, 0, 0], color='k'
      )
  )
  ax.text(-0.075, -0.05, '0', font={'size': 12})
  ax.text(1.025, -0.05, '1', font={'size': 12})
  ax.text(0.475, np.sqrt(3) / 2 + 0.05, '2', font={'size': 12})
  plt.tricontourf(points[:, 0], points[:, 1], labels)
  ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
  ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.spines['bottom'].set_visible(False)
  ax.spines['left'].set_visible(False)
  ax.set_xlim(-0.1, 1.1)
  ax.set_ylim(-0.1, 1.1)
  plt.gcf().set_size_inches(3, 2)
  ax.set_facecolor('white')
  plt.figure(facecolor='white')


def project_simplex(points):
  """Project 3D distributions to the 3-simplex.

  Args:
    points: `num_examples x 3` shaped array of 3D points; usually corresponding
      to distributions over 3 classes to plot on the 3-simplex using
      `plot_simplex`.

  Returns:
    Projected 2D points on the 3-simplex.
  """
  x = 1.0 / 2
  y = 1.0 / (2 * np.sqrt(3))
  x = x - (1.0 / np.sqrt(3)) * points[:, 0] * np.cos(np.pi / 6.0)
  y = y - (1.0 / np.sqrt(3)) * points[:, 0] * np.sin(np.pi / 6.0)
  x = x + (1.0 / np.sqrt(3)) * points[:, 1] * np.cos(np.pi / 6.0)
  y = y - (1.0 / np.sqrt(3)) * points[:, 1] * np.sin(np.pi / 6.0)
  y = y + (1.0 / np.sqrt(3)) * points[:, 2]
  return np.vstack((x, y)).T


def _plot_boundary(points, model_fn):
  """Adds a classifier boundary to a data plot."""
  num_classes = points.shape[1]
  x1grid = np.arange(
      np.min(points[:, 0]) - 0.1, np.max(points[:, 0]) + 0.1, 0.01
  )
  x2grid = np.arange(
      np.min(points[:, 1]) - 0.1, np.max(points[:, 1]) + 0.1, 0.01
  )
  xx, yy = np.meshgrid(x1grid, x2grid)
  r1, r2 = xx.flatten(), yy.flatten()
  r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
  grid = np.hstack((r1, r2))
  yhat = model_fn(grid)
  for label in range(num_classes):
    zz = np.argmax(yhat, axis=1) == label
    zz = zz.reshape(xx.shape)
    plt.contour(xx, yy, zz, [0.5], cmap='viridis')


def _plot_points(points):
  """Adds highlighted points to a data plot."""
  plt.scatter(points[:, 0], points[:, 1], c='black', s=80, marker='x', alpha=1)
  for n in range(points.shape[0]):
    plt.text(
        points[n, 0] + 0.025,
        points[n, 1] + 0.025,
        n,
        fontdict={'fontsize': 12, 'fontweight': 'bold'},
    )


def _annotate_data(**kwargs):
  """Annotates a data plot with axis, legend and title."""
  xlabel = kwargs.get('xlabel', 'x0')
  if xlabel:
    plt.xlabel(xlabel)
  ylabel = kwargs.get('ylabel', 'x1')
  if ylabel:
    plt.ylabel(ylabel)
  legend = kwargs.get('legend', True)
  if legend:
    plt.legend()
  title = kwargs.get('title', 'Examples with their smooth labels')
  if title:
    plt.title(title)
  plt.legend()
  plt.gcf().set_size_inches(kwargs.get('width', 5), kwargs.get('height', 3))
  if kwargs.get('name', False):
    plt.savefig(kwargs.get('name') + '.pdf', bbox_inches='tight')


def plot_smooth_data(
    points, smooth_labels, highlight_points=None, model_fn=None, **kwargs
):
  """Plots 2D data with smooth labels, highlighted points and model boundary.

  Args:
    points: `num_examples x 2` shaped array of data points to plot.
    smooth_labels: `num_examples x num_classes` shaped array of corresponding
      smooth class labels, i.e., distributions.
    highlight_points: `num_highlight_examples x 2` shaped arrays of optional
      data points to highlight.
    model_fn: Optional model function taking a `num_examples x 2` arrays as
      input and returning softmax probabilities of shape `num_examples x
      num_classes` to plot classifier boundary.
    **kwargs: Additional keyword arguments to set axis label and plot title.
  """
  if model_fn is not None:
    _plot_boundary(points, model_fn)
  colors = kwargs.get('colors', COLORS)
  colors = np.dot(smooth_labels, colors)
  plt.scatter(
      points[:, 0], points[:, 1], c=colors, s=kwargs.get('s', 40), alpha=0.6
  )
  if highlight_points is not None:
    _plot_points(highlight_points)
  _annotate_data(**kwargs)


def plot_data(points, labels, highlight_points=None, model_fn=None, **kwargs):
  """Plots 2D data with class labels, highlighted points and model boundary.

  Args:
    points: `num_examples x 2` shaped array of data points to plot.
    labels: `num_examples` shaped array of corresponding class labels.
    highlight_points: `num_highlight_examples x 2` shaped arrays of optional
      data points to highlight.
    model_fn: Optional model function taking a `num_examples x 2` arrays as
      input and returning softmax probabilities of shape `num_examples x
      num_classes` to plot classifier boundary.
    **kwargs: Additional keyword arguments to set axis label and plot title.
  """
  num_classes = np.max(labels) + 1
  if model_fn is not None:
    _plot_boundary(points, model_fn)
  colors = kwargs.get('colors', COLORS)
  for k in range(num_classes):
    plt.scatter(
        points[labels == k, 0],
        points[labels == k, 1],
        c=colors[k],
        label=f'Class {k}',
        s=kwargs.get('s', 40),
        alpha=0.6,
    )
  if highlight_points is not None:
    _plot_points(highlight_points)
  _annotate_data(**kwargs)
