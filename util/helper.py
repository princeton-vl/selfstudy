"""Home to miscellaneous utility functions."""

import numpy as np
import os
import ray
from skimage import color
import sys
import time
import torch


FREE_DELAY_S = 20.0
MAX_FREE_QUEUE_SIZE = 200
MAX_PREFETCH_QUEUE_SIZE = 5
_last_free_time = 0.0
_to_free = []


def ray_get_and_free(object_ids):
  """Call ray.get and then queue the object ids for deletion.
  This function should be used whenever possible in RLlib, to optimize
  memory usage. The only exception is when an object_id is shared among
  multiple readers.
  (this is copied directly from rllib:
  https://github.com/ray-project/ray/blob/master/python/ray/rllib/utils/memory.py)
  Args:
  object_ids (ObjectID|List[ObjectID]): Object ids to fetch and free.
  Returns:
  The result of ray.get(object_ids).
  """

  global _last_free_time
  global _to_free

  result = ray.get(object_ids)
  if type(object_ids) is not list:
    object_ids = [object_ids]
  _to_free.extend(object_ids)

  # batch calls to free to reduce overheads
  now = time.time()
  if (len(_to_free) > MAX_FREE_QUEUE_SIZE
      or now - _last_free_time > FREE_DELAY_S):
    ray.internal.free(_to_free)
    _to_free = []
    _last_free_time = now

  return result


def ray_trial_str_creator(trial):
  """Cleaner Ray experiment names (more or less)."""
  trial_str = 'e'
  for k in trial.config:
    if k not in ['flags', 'loader']:
      trial_str += '_{}={}'.format(k, trial.config[k])
  return trial_str


def suppress_output():
  f = open(os.devnull,'w')
  sys.stdout = f
  sys.stderr = f


def mkdir_p(dirname):
    try: os.makedirs(dirname)
    except FileExistsError: pass


def running_mean(log, k, v, p=.95):
  """Update running average of a given training metric."""

  if not k in log:
    log[k] = v
  else:
    if torch.isnan(torch.tensor(log[k])):
      log[k] = v
    elif not torch.isnan(torch.tensor(v)):
      log[k] = p * log[k] + (1-p) * v

  return log[k]


class RGB2Lab(object):
  """Convert RGB PIL image to ndarray Lab."""
  def __call__(self, img):
    img = color.rgb2lab(img)
    return torch.FloatTensor(img).permute(2,0,1)


def get_train_val_idxs(split, labels, n_valid, n_classes,
                       semisupervised=None, min_len=10000):
  """Set up training/validation split. """
  dataset_size = len(labels)
  n_train = dataset_size - n_valid
  if split == 'train':
    idxs = np.arange(n_train)

    if semisupervised is not None:
      if labels.ndim < 4:
        # Ensure that an equal number of each class is included
        # Should probably instead preserve proportions from original labels
        labels = np.array(labels)[idxs]
        per_class = semisupervised // n_classes
        class_idxs = [idxs[labels == i] for i in range(n_classes)]
        idxs = np.concatenate([c[:per_class] for c in class_idxs])
      else:
        idxs = idxs[:semisupervised]

      if len(idxs) < min_len:
        to_repeat = int(np.ceil(min_len / len(idxs)))
        idxs = np.concatenate([idxs] * to_repeat)

  elif split == 'valid':
    idxs = np.arange(n_valid) + n_train

  return np.array(idxs).astype(int)


def define_lr_vals(milestones, target_vals, iters_per_epoch, exp_factor=0):
  lr_vals = []
  if not isinstance(target_vals, list):
    target_vals = [0, target_vals, 0]
  for i in range(len(milestones)):
    i0 = 0 if i == 0 else milestones[i-1]
    i1 = milestones[i]
    t0,t1 = target_vals[i:i+2]

    tmp_vals = np.linspace(t0, t1, iters_per_epoch*(i1-i0))
    if i == len(milestones) - 1:
      e = np.linspace(1, 1e-7, iters_per_epoch*(i1-i0)) ** exp_factor
      tmp_vals *= e

    lr_vals += list(tmp_vals)
  return lr_vals


def create_filter(arr, conditions):
  init_filt = np.ones(len(arr)).astype(bool)
  for c in conditions:
    init_filt *= c
  return np.arange(len(arr))[init_filt]


def get_pose_class_idx(rot_mat):
  r = rot_mat[:,:,1]
  theta = torch.atan2((r[:,0]**2 + r[:,1]**2)**.5, r[:,2])
  psi = torch.atan2(r[:,0], r[:,1])

  is_top = theta < (np.pi / 8)
  c1 = (~is_top) & ((psi < -3*np.pi/4) | (psi >= 3*np.pi/4))
  c2 = (~is_top) & (psi < -np.pi/4) & (psi >= -3*np.pi/4)
  c3 = (~is_top) & (psi < np.pi/4) & (psi >= -np.pi/4)
  c4 = (~is_top) & (psi < 3*np.pi/4) & (psi >= np.pi/4)
  tmp_targets = torch.LongTensor(len(r))
  tmp_targets[is_top] = 0
  tmp_targets[c1] = 1
  tmp_targets[c2] = 2
  tmp_targets[c3] = 3
  tmp_targets[c4] = 4

  return tmp_targets
