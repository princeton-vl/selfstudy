"""Functions for managing and visualizing experiment results."""

from copy import deepcopy
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns

from PIL import Image, ImageDraw, ImageFont


misc_ref = {}


def convert_ds_name(ds):
  """Convert from dataset ID to more easily understood string."""
  ref_name = ['T','C','V','L']

  if isinstance(ds, str):
    ds = ds[-4:]
  else:
    ds = f'{int(ds):04d}'

  for c_idx, c in enumerate(ds):
    if c == '0':
      ref_name[c_idx] = '-'

  return ''.join(ref_name)


def merge_results_aux(d1, d2):
  # Leaf nodes expected to be numpy arrays
  assert isinstance(d1, dict), print('Inputs to merge_aux expected to be dictionaries.')

  # Iterate through keys in first dict, updating with info from second
  for k in d1:
    if k in d2:
      if isinstance(d1[k], dict):
        merge_results_aux(d1[k], d2[k])
      else:
        d1[k] = np.concatenate([d1[k],d2[k]], 0)

  for k in d2:
    # Add any missing keys from second dict
    if not k in d1:
      d1[k] = deepcopy(d2[k])


def merge_results(to_merge):
  """Merge list of organized results

  Values should be those after calling `organize_as_dict` given a particular
  experiment. All results to be merged are expected to be organized same way.
  """
  new_r = deepcopy(to_merge[0])
  for r in to_merge[1:]:
    merge_results_aux(new_r, r)

  return new_r


# ==============================================================================
# Class for loading/organizing experiment results
# ==============================================================================

class ExperimentResult:
  """Class for managing results from experiments configured and run with ray."""

  def __init__(self, path):
    self.path = path

    self.update_results()
    self.process_params()

  def update_results(self, get_test=False):
    """Load all results from associated experiment directory."""

    # Assumes all experiments in directory use same config
    # (behavior undefined otherwise)
    k_ref = None
    self.results_raw = []
    self.params_raw = []
    self.exp_lens = []

    # Scan directory for experiment folders
    exp_folders = os.listdir(self.path)
    exp_folders = [e for e in exp_folders if e[:2] == 'e_']

    # Iterate through and parse results
    for e in exp_folders:
      try:
        with open(f'{self.path}/{e}/result.json', 'r') as f:
          r = [json.loads(line) for line in f]

        if len(r) > 0:
          if k_ref is None:
            k_ref = [k for k in r[0]['config'] if k not in ['flags']]
            p_keys = [k.split('.')[-1] for k in k_ref]

          self.results_raw += [r]
          self.exp_lens += [len(r)]
          self.params_raw += [[r[0]['config'][k] for k in k_ref]]

      except:
        pass

    # Read last reported accuracy from each experiment
    train_acc = np.array([r[-1]['train_accuracy'] for r in self.results_raw])
    try:
      split = 'valid' if not get_test else 'test'
      valid_acc = np.array([r[-1][f'{split}_accuracy'] for r in self.results_raw])
      acc = np.stack([train_acc, valid_acc], 1)
    except:
      acc = train_acc[:,None]

    self.acc = acc * 100
    self.params_raw = np.array(self.params_raw)
    self.param_keys = p_keys

  def process_params(self,
                     to_ignore=['pretrained', 'model_choice'],
                     custom_rules=None):
    """Update params to be easier to work with."""

    global misc_ref
    params = self.params_raw.copy() # Don't change original values
    p_keys = self.param_keys

    # Remove any accidental duplicate keys
    seen_keys, to_remove = [], []
    for k_idx, k in enumerate(p_keys):
      if not k in seen_keys:
        seen_keys += [k]
      else:
        to_remove = [k_idx] + to_remove

    for k_idx in to_remove:
      params = np.delete(params, k_idx, axis=1)
      del p_keys[k_idx]

    # Remove any keys that should be ignored
    for k in to_ignore:
      if k in p_keys:
        tmp_idx = p_keys.index(k)
        params = np.delete(params, tmp_idx, axis=1)
        del p_keys[tmp_idx]

    # Manage any custom processing
    to_update = {
      'dataset_choice': lambda x: int(x[-4:]),
      'milestones': lambda x: x[-1],
      'target_lrs': lambda x: x[1] if isinstance(x, list) else x,
      'resnet_choice': lambda x: int(x[6:]),
    }

    if custom_rules is not None:
      for k, v in custom_rules.items():
        to_update[k] = v

    for k, update_fn in to_update.items():
      if k in p_keys:
        idx = p_keys.index(k)
        for p in params:
          p[idx] = update_fn(p[idx])

    if 'misc' in p_keys:
      idx = p_keys.index('misc')

      if isinstance(params[0,idx], str):
        if params[0,idx].split('-')[-1].isnumeric():
          p_keys += ['pretrain_amt']
          params = np.concatenate([params, np.zeros((len(params),1))],1)

        for p in params:
          split_misc = p[idx].split('-')
          tmp_misc = split_misc[0]
          if 'pretrain_amt' in p_keys:
            p[-1] = split_misc[-1]

          if tmp_misc not in misc_ref:
            next_val = max([k for k in misc_ref.keys() if isinstance(k, int)]) + 1
            misc_ref[tmp_misc] = next_val
            misc_ref[next_val] = tmp_misc

          p[idx] = misc_ref[tmp_misc]

    try:
      self.params = params.astype(float)
    except:
      print("Error occurred when converting params to array")
      print(p_keys)
      print(params)

  def organize_as_dict(self, exp_data=None, key_list=['dataset_choice', 'semisupervised'],
                       filt_idxs=None):
    if exp_data is None:
      params, acc, p_keys = self.params, self.acc, self.param_keys
    else:
      params, acc, p_keys = exp_data

    if filt_idxs is not None:
      params = params[filt_idxs]
      acc = acc[filt_idxs]

    result_dict = {}

    if key_list is None:
      key_list = p_keys

    if len(key_list) == 0:
      return acc

    k_idx = p_keys.index(key_list[0])
    k_vals, idx_ref = np.unique(params[:,k_idx], return_inverse=True)
    k_vals.sort()

    for i, k_val in enumerate(k_vals):
      tmp_params = params[idx_ref == i]
      tmp_acc = acc[idx_ref == i]

      if key_list[0] == 'dataset_choice':
        k_val = convert_ds_name(k_val)
      if key_list[0] == 'misc':
        k_val = misc_ref[k_val]

      result_dict[k_val] = self.organize_as_dict((tmp_params, tmp_acc, p_keys),
                                                 key_list=key_list[1:])

    return result_dict

  def organize_as_arr(self, exp_data=None, key_list=None):
    if exp_data is None:
      params, acc, p_keys = self.params, self.acc, self.param_keys
    else:
      params, acc, p_keys = exp_data

    result_mean = []
    result_std = []

    if key_list is None:
      key_list = p_keys

    if len(key_list) == 0:
      return acc.mean(0), acc.std(0), []

    k_idx = p_keys.index(key_list[0])
    k_vals, idx_ref = np.unique(params[:,k_idx], return_inverse=True)
    k_vals.sort()

    k_vals = list(k_vals)

    for i, k_val in enumerate(k_vals):
      tmp_params = params[idx_ref == i]
      tmp_acc = acc[idx_ref == i]

      if key_list[0] == 'dataset_choice':
        k_vals[i] = convert_ds_name(k_val)

      m, s, curr_keys = self.organize_as_arr((tmp_params, tmp_acc, p_keys),
                                             key_list=key_list[1:])
      result_mean += [m]
      result_std += [s]

    return np.stack(result_mean), np.stack(result_std), [k_vals] + curr_keys


# ==============================================================================
# Plotting functions
# ==============================================================================

def get_n_rows_cols(axs):
  if isinstance(axs, np.ndarray):
    return axs.shape
  else:
    return 1, 1

d_amt_str_ref = {100:'100', 250:'250', 1000:'1e3', 4000:'4e3', 20000:'2e4',
                 50000:'5e4', 100000:'1e5', 200000:'2e5'}

def ax_setup(ax, d, title=None, include_legend=True,
             min_y=0, max_y=100, min_x=100, max_x=500000,
             ylabel='accuracy (%)', xlabel='# labeled samples',
             legend_loc='lower right'):
  ax.set_xscale('log')
  tmp_labels = ['a' for d_ in d] #d_amt_str_ref[d_] for d_ in d]
  # ax.set_xticks(list(d)) #, tmp_labels)
  # ax.set_xticklabels(tmp_labels,rotation=0)
  # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
  ax.set_xlabel(xlabel)
  ax.set_ylabel(ylabel)
  ax.set_xlim(min_x, max_x)
  ax.set_ylim(min_y, max_y)

  if include_legend: ax.legend(loc=legend_loc, fontsize=9)
  if title is not None: ax.set_title(title);


def plot_errorbar(ax, vals, d_amts=None, label='', is_valid=1,
                  label_split=False, **kargs):
  # x-vals
  if d_amts is None:
    d_amts = list(vals.keys())
    d_amts.sort()

  # y-vals (mean/std)
  y_mean = np.array([vals[d][:, is_valid].mean() for d in d_amts])
  y_std = np.array([vals[d][:, is_valid].std() for d in d_amts])

  # label
  split_str = (' (valid)' if is_valid else ' (train)') if label_split else ''
  label_str = label + split_str
  ax.errorbar(d_amts, y_mean, y_std, label=label_str, **kargs)


def subplot_choice(axs, ax_idx):
  n_rows, n_cols = get_n_rows_cols(axs)

  if n_rows == 1 and n_cols == 1:
    tmp_ax = axs
  else:
    if isinstance(ax_idx, int):
      tmp_ax = axs[ax_idx // n_cols][ax_idx % n_cols]
    else:
      tmp_ax = axs[ax_idx[0]][ax_idx[1]]

  return tmp_ax


def compare_training_curves(results, params, to_compare, filters,
                            labels=None, filt_labels=None, splits=['train', 'valid'],
                            n_cols=None, colors=None, tight_layout=True, y_max=1,
                            label_mapping=None):
  n_exps = len(results)
  n_plots = len(filters) * len(splits)
  if n_cols is None: n_cols = len(filters)
  n_rows = int(np.ceil(n_plots / n_cols))

  f = plt.figure(figsize=[4*n_cols, 4*n_rows])
  count = 1

  for filt_idx, filt in enumerate(filters):
    valid_exps = np.arange(n_exps)[filt]

    for split_idx, split in enumerate(splits):
      ax = f.add_subplot(n_rows, n_cols, count)
      labeled = [False] * len(np.unique(params[:,to_compare]))

      for exp_idx in valid_exps:
        acc_curve = [r_[f'{split}_accuracy'] for r_ in results[exp_idx]]
        p_idx = params[exp_idx, to_compare]
        if label_mapping: p_idx = label_mapping[p_idx]
        p_idx = int(p_idx)

        label = None
        if labels and not labeled[p_idx]:
          label = labels[p_idx]
          labeled[p_idx] = True

        c = colors[p_idx] if colors else None
        ax.plot(acc_curve, c=c, label=label)

      ax.set_ylim(0, y_max)
      if filt_labels:
        ax.set_title(f'{filt_labels[filt_idx]} ({split})')

      if labels and count == 1:
        ax.legend()

      count += 1

  if tight_layout:
    f.tight_layout()


def linear_interpolate_x(x0, x1, y0, y1, y):
  assert y0 <= y <= y1, f'y ({y}) not between ({y0}, {y1})'
  frac = (y - y0) / (y1 - y0)
  return x0 + (x1 - x0) * frac


def get_x_val(r, y):
  tmp_amts = list(r.keys())
  tmp_amts.sort()

  for amt_idx, amt in enumerate(tmp_amts):
    if r[amt][:,1].mean() > y:
      if amt_idx == 0:
        return 0
      else:
        x0,x1 = tmp_amts[amt_idx-1], amt
        y0,y1 = r[x0][:,1].mean(), r[x1][:,1].mean()
        return linear_interpolate_x(x0, x1, y0, y1, y)

  return -1


def calc_utility(v0, v1, d_amts=None):
  if d_amts is None:
    d_amts = list(v1.keys())
    d_amts.sort()

  ratios = []
  ratio_std = []
  final_d_amts = []
  y_prev, d_prev = 0, 0

  max_d_amt = max(v0, key=lambda k: v0[k][:,1].mean())
  max_val = v0[max_d_amt][:,1].mean()

  for d_amt in d_amts:
    if d_amt in v1:
      y = v1[d_amt][:,1].mean()

      if y > max_val:
        max_x = linear_interpolate_x(d_prev, d_amt, y_prev, y, max_val)
        ratios += [max_d_amt / max_x]
        ratio_std += [0]
        final_d_amts += [max_x]
        break

      else:

        tmp_r = []
        for y_ in v1[d_amt][:,1]:
          x_ = get_x_val(v0, y_)
          tmp_r += [x_ / d_amt - 1]
        tmp_r = np.array(tmp_r)

        ratios += [tmp_r.mean()]
        ratio_std += [tmp_r.std()]
        # ratios += [x_ / d_amt - 1 if not x_ == -1 else 10000]
        final_d_amts += [d_amt]

        #x_ = get_x_val(v0, y)
        #ratios += [x_ / d_amt - 1 if not x_ == -1 else 10000]

      y_prev, d_prev = y, d_amt

  return final_d_amts, ratios, ratio_std


def result_plot(use_markers=True, min_y=0, max_y=100, legend_loc='lower right',
                xlabel='# labeled samples', ylabel='Accuracy (%)'):
  def plot_decorator(plot_fn):
    def wrapper(results, to_compare,
                ds_names=None, d_amts=None, include_legend=True,
                ax_ref=None, n_rows=1, n_cols=2, figsize=[8,4],
                c_ref=None, l_ref=None, m_ref=None,
                m_size=5, use_markers=use_markers, legend_loc=legend_loc,
                min_y=min_y, max_y=max_y, xlabel=xlabel, ylabel=ylabel,
                **kargs):
      # Which datasets to display in figure
      ds_names = ds_names or list(results[to_compare[0]].keys())

      if d_amts is None:
        d_amts = list(results[to_compare[0]][ds_names[0]].keys())
        d_amts.sort()

      # Set up subplots
      if ax_ref is None:
        f, axs = plt.subplots(n_rows, n_cols, figsize=figsize,
                              dpi=200, squeeze=False)
      else:
        axs = ax_ref
        n_rows, n_cols = get_n_rows_cols(axs)

      assert len(ds_names) <= (n_rows * n_cols), \
        f'Figure is {n_rows}x{n_cols} but {len(ds_names)} datasets provided.'

      # Color + marker defaults
      c_ref = c_ref or sns.color_palette()
      m_ref = m_ref or ['*', 'o', 's', '^', 'D', 'x', 'd']

      all_data = {}

      # Prepare subplot for each dataset
      for ds_idx, ds in enumerate(ds_names):
        ax = subplot_choice(axs, ds_idx)
        plot_data = plot_fn(ax, ds, results, to_compare, d_amts,
                            c_ref, l_ref, m_ref, m_size, use_markers, max_y,
                            **kargs)
        all_data[ds] = plot_data

        tmp_xlabel = xlabel if ds_idx // n_cols == n_rows - 1 else None
        tmp_ylabel = ylabel if ds_idx % n_cols == 0 else None

        ax_setup(ax, d_amts, 'Dataset: ' + ds,
                 xlabel=tmp_xlabel, ylabel=tmp_ylabel,
                 min_x=min(d_amts)-10, max_x=max(d_amts),
                 min_y=min_y, max_y=max_y,
                 include_legend=(ds_idx==0 and include_legend),
                 legend_loc=legend_loc)

      if ax_ref is None:
        f.tight_layout(rect=[0, 0.03, 1, 0.95])

      return all_data

    return wrapper
  return plot_decorator


@result_plot()
def accuracy_plot(ax, ds, results, to_compare,
                  d_amts, c_ref, l_ref, m_ref, m_size, use_markers, max_y,
                  include_train=False):
  # Plot training and validation accuracy
  for alg_idx, alg in enumerate(to_compare):
    plt_kargs = {'c':c_ref[alg_idx], 'linewidth':1.5}
    if use_markers:
      plt_kargs['marker'] = m_ref[alg_idx]
      plt_kargs['markersize'] = m_size
    if l_ref is not None:
      plt_kargs['ls'] = l_ref[alg_idx]

    if include_train:
      plot_errorbar(ax, results[alg][ds], label='', is_valid=0, ls='--', **plt_kargs)
    plot_errorbar(ax, results[alg][ds], label=alg, **plt_kargs)


@result_plot(use_markers=True, max_y=10, ylabel='Utility', legend_loc='upper right')
def utility_plot(ax, ds, results, to_compare,
                 d_amts, c_ref, l_ref, m_ref, m_size, use_markers, max_y,
                 baseline='scratch'):
  """Visualize utility results."""
  utility_results = {}

  for alg_idx, alg in enumerate(to_compare):
    # Baseline to compare utility against
    b_key = baseline if isinstance(baseline, str) else baseline[alg_idx]
    base = results[b_key]

    # Compute utility values
    tmp_d_amts, u, u_std = calc_utility(base[ds], results[alg][ds], d_amts=d_amts)
    utility_results[alg] = [tmp_d_amts, u]

    # Plot result
    plt_kargs = {'label':alg, 'c':c_ref[alg_idx + 1], 'linewidth':1.5}
    if use_markers:
      plt_kargs['marker'] = m_ref[alg_idx + 1]
      plt_kargs['markersize'] = m_size
    if l_ref is not None:
      plt_kargs['ls'] = l_ref[alg_idx]

    ax.errorbar(tmp_d_amts, u, u_std, **plt_kargs)
    # ax.plot(tmp_d_amts, u, **plt_kargs)

  # Visualize region where we can appropriately calculate utility
  if isinstance(baseline, str):
    base = results[baseline][ds]
    max_d_amt = max(base, key=lambda k: base[k][:,1].mean())
    max_utility = max_d_amt / np.array(d_amts)
    ax.plot(d_amts, max_utility, ls='--', c='gray', linewidth=2)
    ax.fill_between(d_amts, max_utility, np.ones(len(max_utility)) * max_y,
                    interpolate=True, color='gray', alpha=.3)

  return utility_results


# ==============================================================================
# Hyperparam comparison
# (Haven't checked this recently, might be broken. Decided to keep it in just in
# case it proves useful to someone.)
# ==============================================================================

def plot_hyperparam_comparison(params, acc, log_ref, key_ref,
                               filter_idxs=None, cmap=None):
  if filter_idxs is None: filter_idxs = np.arange(len(params))
  acc = np.array(acc)[filter_idxs]
  n_params = params.shape[1]

  f = plt.figure(figsize=(16,16))
  count = 1
  for p_idx_1 in range(n_params - 1):
    for p_idx_2 in range(p_idx_1+1, n_params):
      if p_idx_1 != p_idx_2:
        ax = f.add_subplot(n_params, n_params, count)
        ax.scatter(params[filter_idxs,p_idx_1],
                   params[filter_idxs,p_idx_2],
                   c=acc, cmap=cmap)
        if log_ref[p_idx_1]: ax.set_xscale('log')
        if log_ref[p_idx_2]: ax.set_yscale('log')

        ax.set_xlim(params[:,p_idx_1].min(), params[:,p_idx_1].max())
        ax.set_ylim(params[:,p_idx_2].min(), params[:,p_idx_2].max())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.minorticks_off()
        ax.set_xlabel(key_ref[p_idx_1])
        ax.set_ylabel(key_ref[p_idx_2])
        count += 1


def min_removal(acc, idx_ref, n):
  # This is used to remove outliers (comes up a bit in hyperparam search)
  new_idxs = idx_ref.copy()
  for i in range(n):
    min_idx = acc[new_idxs].argmin()
    new_idxs = np.concatenate([new_idxs[:min_idx], new_idxs[min_idx+1:]])

  return new_idxs


def line(img, pt1, pt2, color, width, is_draw=False):
  """Draw a line on an image."""

  # Make sure dimension of color matches number of channels in img
  pt1 = np.array(pt1,dtype=int) if type(pt1) == list else pt1.astype(int)
  pt2 = np.array(pt2,dtype=int) if type(pt2) == list else pt2.astype(int)

  if is_draw:
    img.line((pt1[0],pt1[1],pt2[0],pt2[1]), fill=color, width=width)
    return

  else:
    tmp_img = Image.fromarray(img)
    tmp_draw = ImageDraw.Draw(tmp_img)
    tmp_draw.line((pt1[0],pt1[1],pt2[0],pt2[1]), fill=color, width=width)
    return np.array(tmp_img)


def draw_text(img, pt, text, color=(0,0,0), draw_bg=False, is_draw=False):
  """Draw text on image at specified coordinate."""

  # Round the coordinate location down
  pt = pt.astype(int)

  if is_draw:
    if draw_bg:
        img.rectangle([pt[0],pt[1],pt[0]+7*len(text),pt[1]+15],fill=(255,255,255))
    img.text([pt[0]+3,pt[1]+3], text, color)
    return

  else:
    tmp_img = Image.fromarray(img)
    tmp_draw = ImageDraw.Draw(tmp_img)
    if draw_bg:
        tmp_draw.rectangle([pt[0],pt[1],pt[0]+7*len(text),pt[1]+15],fill=(255,255,255))
    tmp_draw.text([pt[0]+3,pt[1]+3], text, color)
    return np.array(tmp_img)


def draw_number_grid(arr, number_mode='default', precision=2, err=None):
  """Creates an image displaying a 2D array of numbers."""

  cell_ht = 50
  cell_wd = 150
  n_rows, n_cols = arr.shape
  ht, wd = n_rows * cell_ht, n_cols * cell_wd

  tmp_img = np.ones((ht, wd, 3)).astype(np.uint8) * 255
  # for row_idx in range(n_rows):
  #   if row_idx % 2 == 1:
  #     tmp_img[row_idx*cell_ht+2:(row_idx+1)*cell_ht+2,:] = [255, 242, 250]

  tmp_img = Image.fromarray(tmp_img)


  tmp_draw = ImageDraw.Draw(tmp_img)

  for row_idx in range(n_rows):
    line(tmp_draw, [0, row_idx * cell_ht + 2], [wd, row_idx * cell_ht + 2], (0,0,0), 2, True)
    for col_idx in range(n_cols):
      # if row_idx == 0:
      #   line(tmp_draw, [col_idx * cell_wd, 0], [col_idx * cell_wd, ht], (0,0,0), 4, True)
      y_idx = row_idx * cell_ht + 10
      x_idx = col_idx * cell_wd + cell_wd // 10
      val = arr[row_idx, col_idx]

      if number_mode == 'default':
        num_str = '%d' % val
      elif number_mode == 'hex':
        num_str = '%X' % val
      elif number_mode == 'float':
        num_str = f'{val:.{precision}f}'
        if err is not None:
          num_str += f' ±{err[row_idx, col_idx]:.{precision}f}'

        # if num_str[0] != '-':
        #   num_str = num_str[1:]
        # else:
        #   num_str = '-' + num_str[2:]
      else:
        raise ValueError("Whoops, unsupported number mode. Try 'default', 'hex', or 'float'.")

      draw_text(tmp_draw, np.array([x_idx, y_idx]), num_str, is_draw=True)

  line(tmp_draw, [0, (row_idx+1) * cell_ht - 1], [wd, (row_idx+1) * cell_ht - 1], (0,0,0), 4, True)

  return np.array(tmp_img)


def display_hyperparam_results(p_vals, p_acc, p_std, p_keys,
                               plt_idx=2, x_idx=0, y_idx=1, plot_valid=1,
                               disp_text=True, precision=1, n_rows=2,
                               plt_wd=4, plt_ht=4, tight_layout=True):

  tmp_acc = p_acc[..., plot_valid].transpose(plt_idx, y_idx, x_idx) * 100
  tmp_std = p_std[..., plot_valid].transpose(plt_idx, y_idx, x_idx) * 100
  p_val_str = lambda x: ", ".join([str(p_) for p_ in x])

  n_cols = int(np.ceil(len(p_vals[plt_idx]) / n_rows))
  f = plt.figure(figsize=(plt_wd*n_cols, plt_ht*n_rows))
  count = 1

  for p_idx, p_im in enumerate(tmp_acc):
    disp_im = (p_im - p_im.min()) / (p_im.max() - p_im.min())
    y_tick_range = np.arange(p_im.shape[0])
    x_tick_range = np.arange(p_im.shape[1])

    if disp_text:
      disp_im = draw_number_grid(p_im, number_mode='float',
                                 precision=precision, err=tmp_std[p_idx])
      cell_ht = disp_im.shape[0] // p_im.shape[0]
      cell_wd = disp_im.shape[1] // p_im.shape[1]
      y_tick_range = y_tick_range * cell_ht + cell_ht / 2
      x_tick_range = x_tick_range * cell_wd + cell_wd / 2

    ax = f.add_subplot(n_rows, n_cols, count)
    ax.imshow(disp_im, vmin=0, vmax=1)
    ax.set_title(f'{p_keys[plt_idx]}: {p_val_str(p_vals[plt_idx][p_idx])}')

    ax.set_yticks(y_tick_range)
    ax.set_yticklabels([f'{p_val_str(p)}' for p in p_vals[y_idx]])
    ax.set_ylabel(p_keys[y_idx])

    ax.set_xticks(x_tick_range)
    ax.set_xticklabels([f'{p_val_str(p)}' for p in p_vals[x_idx]], rotation=0)
    ax.set_xlabel(p_keys[x_idx])

    ax.grid(b=None)

    count += 1

  if tight_layout:
    f.tight_layout()


def str_fn(x,r):
  if not (isinstance(x, list) or isinstance(x, np.ndarray)):
    return f'{x:{r}}'
  elif len(x) == 1:
    return f'{x[0]:{r}}'
  else:
    tmp_str = ', '.join(f'{x_:{r}}' for x_ in x)
    return f'[{tmp_str}]'


def compare_hyper_exps(p_vals, p_acc, p_std, p_keys,
                       idx, map_ref, control_idxs={},
                       disp_text=True, precision=1,
                       ax=None, plt_wd=4, plt_ht=4):

  # Add train/valid dimension
  all_keys = p_keys + ['is_valid']
  all_pvals = p_vals + [[0.,1.]]

  if isinstance(idx, str):
    idx = all_keys.index(idx)

  # Map keys and values from control_idxs to indices
  ctl_k = list(control_idxs.keys())
  ctl_v = list(control_idxs.values())
  ctl_idxs, ctl_vals = [], []
  for i, k in enumerate(ctl_k):
    ctl_idxs += [all_keys.index(k)]
    ctl_vals += [all_pvals[ctl_idxs[-1]].index(ctl_v[i])]

  # Setup plot title
  if not control_idxs:
    plt_title = 'All experiments'
  else:
    t_str = []
    for i, t in enumerate(ctl_idxs):
      p_str = str_fn(all_pvals[t][ctl_vals[i]],map_ref[t])
      t_str += [f'{all_keys[t]}: {p_str}']
    plt_title = 'Fixed params: ' + ', '.join(t_str)

  # Rearrange results array
  n_dims = p_acc.ndim
  other_idxs = np.setdiff1d(np.arange(n_dims), [idx] + ctl_idxs)
  new_order = list(other_idxs) + [idx] + ctl_idxs

  tmp_acc = p_acc.transpose(*new_order)
  for v in ctl_vals[::-1]:
    tmp_acc = tmp_acc[...,v]
  tmp_acc = tmp_acc.reshape(-1, len(p_vals[idx]))

  # Do the same for std
  tmp_std = p_std.transpose(*new_order)
  for v in ctl_vals[::-1]:
    tmp_std = tmp_std[...,v]
  tmp_std = tmp_std.reshape(-1, len(p_vals[idx]))

  # Set up experiment list
  tmp_keys = [all_keys[i] for i in other_idxs]
  tmp_pvals = [all_pvals[i] for i in other_idxs]
  tmp_ref = [map_ref[i] for i in other_idxs]

  p_idxs = np.meshgrid(*[np.arange(len(p)) for p in tmp_pvals], indexing='ij')
  p_idxs = np.stack([p.flatten() for p in p_idxs], 1)

  t_str = ', '.join([f'{k}' for k in tmp_keys])
  e_str = []
  for p in p_idxs:
    p_str = [str_fn(tmp_pvals[i][p[i]], tmp_ref[i]) for i in range(len(p))]
    e_str += [', '.join(p_str)]

  # Display results
  disp_im = (tmp_acc - tmp_acc.min()) / (tmp_acc.max() - tmp_acc.min())
  y_tick_range = np.arange(disp_im.shape[0])
  x_tick_range = np.arange(disp_im.shape[1])

  if disp_text:
    disp_im = draw_number_grid(tmp_acc, number_mode='float',
                               precision=precision, err=tmp_std)
    cell_ht = disp_im.shape[0] // tmp_acc.shape[0]
    cell_wd = disp_im.shape[1] // tmp_acc.shape[1]
    y_tick_range = y_tick_range * cell_ht + cell_ht / 2
    x_tick_range = x_tick_range * cell_wd + cell_wd / 2

  if ax is None:
    f = plt.figure(figsize=(plt_wd, plt_ht))
    ax = f.add_subplot(1,1,1)

  ax.imshow(disp_im, vmin=0, vmax=1)
  ax.set_title(plt_title)

  ax.set_yticks(y_tick_range)
  ax.set_yticklabels(e_str)
  ax.set_ylabel('Params: ' + t_str)

  x_ticks = [f'{str_fn(p,map_ref[idx])}' for p in all_pvals[idx]]
  ax.set_xticks(x_tick_range)
  ax.set_xticklabels(x_ticks, rotation=0)
  ax.set_xlabel(p_keys[idx])

  ax.grid(b=None)
