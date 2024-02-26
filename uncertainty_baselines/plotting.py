# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for building common plots."""

import seaborn as sns


tableau20 = [
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
]
for i in range(len(tableau20)):
  r, g, b = tableau20[i]
  tableau20[i] = (r / 255., g / 255., b / 255.)


def _hue_order_sort_key_fn(s):
  if "(" in s:
    return s.split("(")[1]
  return s


def _get_hue_order(plot_data):
  return sorted(
      plot_data["method"].unique(), reverse=False, key=_hue_order_sort_key_fn)


def shift_level_box_plot(ax,
                         plot_data,
                         y_label,
                         methods_to_colors,
                         legend_loc=None,
                         hue_order=None,
                         fontsize=22,
                         in_distribution_line_width=0.8):
  """Make a boxplot across data splits, grouped by method, per shift level.

  Given `plot_data` (a pd.DataFrame), build a boxplot of metric performance (as
  measured by plot_data["value"]), with a different box per plot_data["method"],
  and a different group of boxes per plot_data["level"]. To also include solid
  lines for the performance on the in-distribution test set, plot_data["level"]
  should include values equal to "Test". See sns.boxplot for more info.

  See Fig. 2 in https://arxiv.org/abs/1906.02530 for an example.

  The 0.8 default for `in_distribution_line_width` comes from here:
  https://github.com/mwaskom/seaborn/blob/536fa2d8e9e8bb098b75174fbd8e2c91967e3b51/seaborn/categorical.py#L2200.
  0.8 is the total width of all the plots at a level combined.

  Args:
    ax: matplotlib Axes to plot on.
    plot_data: a pd.DataFrame with columns "level", "value", and "method", used
      as the data for the box plots.
    y_label: the vertical label for the plot.
    methods_to_colors: an optional dict mapping method string names (values in
      plot_data["method"]) to colors to be used by matplotlib.
    legend_loc: an optional string location for ax.legend. If None, then no
      legend is made.
    hue_order: an optional list of method string names (values in
      plot_data["method"]), the order of boxes in each group (passed to
      sns.boxplot). If None, then the values in plot_data["method"] are
      organized according to `_get_hue_order` defined above.
    fontsize: an int font size for the plot.
    in_distribution_line_width: the total width of all the lines used for the
      in-distribution Test split plot.
  """
  required_keys = ["level", "value", "method"]
  for key in required_keys:
    if key not in plot_data:
      # NOTE(znado): `list(plot_data)` gets a list of column names.
      raise ValueError(
          "{} missing from plot data DataFrame (existing keys: {}).".format(
              key, ",".join(list(plot_data))))

  if hue_order is None:
    hue_order = _get_hue_order(plot_data)

  sns.boxplot(
      x="level",
      y="value",
      ax=ax,
      hue="method",
      data=plot_data[plot_data["level"] != "Test"],
      whis=100.,
      order=["Test", 1, 2, 3, 4, 5],
      hue_order=hue_order,
      palette=methods_to_colors)
  if legend_loc is not None:
    ax.legend(
        ncol=1,
        title="Method",
        framealpha=0.5,
        loc=legend_loc,
        fontsize=fontsize)
  ax.set_xlabel("Shift intensity", fontsize=fontsize)
  ax.set_ylabel(y_label, fontsize=fontsize)
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)
  ax.get_xaxis().tick_bottom()
  ax.get_yaxis().tick_left()

  in_dist_plot_data = plot_data[plot_data["level"] == "Test"]
  if in_dist_plot_data.empty:
    return
  # Plot the in distribution test set (shift level 0) as thick lines instead of
  # box plots.
  x_low = -in_distribution_line_width / 2
  width = in_distribution_line_width / len(in_dist_plot_data)
  for method in hue_order:
    color = methods_to_colors[method]
    value = in_dist_plot_data[
        in_dist_plot_data["method"] == method].value.to_numpy()
    if len(value) == 0:  # pylint: disable=g-explicit-length-test
      continue
    value = value[0]
    ax.plot(
        [x_low, x_low + width],
        [value, value],
        color=color,
        linewidth=4.,
        solid_capstyle="butt")
    x_low += width
