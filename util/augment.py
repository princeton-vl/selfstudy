""" Functions for on-GPU data augmentation.

Slightly modified from: https://myrtle.ai/how-to-train-your-resnet
"""

from collections import namedtuple
import numpy as np
import torch


chunks = lambda data, splits: (data[start:end] for (start, end) in zip(splits, splits[1:]))
even_splits = lambda N, num_chunks: np.cumsum([0] + [(N//num_chunks)+1]*(N % num_chunks) +
                                              [N//num_chunks]*(num_chunks - (N % num_chunks)))


def apply_tf_aux(data, transform, idxs=None, opts=None, max_options=500):
  # Randomly shuffle data
  if idxs is None:
    idxs = torch.randperm(len(data), device=data.device)
  data = data[idxs]

  # Choose subset of options for applying augmentation
  if opts is None:
    opts = shuffled(transform.options(data.shape), inplace=True)[:max_options]

  # Apply transformations to evenly split chunks of the data
  data_chunks = chunks(data, even_splits(len(data), len(opts)))
  data = [transform.apply(x, **tf_args) for x, tf_args in zip(data_chunks, opts)]
  applied_tfs = np.array([[opts[i]] * len(d) for i, d in enumerate(data)])
  applied_tfs = np.concatenate(applied_tfs, 0)
  data = torch.cat(data)

  sorted_idxs = torch.argsort(idxs)
  data = data[sorted_idxs]
  applied_tfs = applied_tfs[sorted_idxs.cpu()]

  return data, applied_tfs


def apply_tf(data, transform, targets=None, transform_targets=False,
             double_sample=False, max_options=None):
  """Applies a transformation to data (and targets if applicable)"""
  n_imgs = len(data)

  # Check whether to apply transformation to targets
  tf_name = transform.__class__.__name__
  img_only_tfs = ['Grayscale', 'ColorJitter', 'Cutout']
  tgts = True if (targets is not None and
                  transform_targets and
                  tf_name not in img_only_tfs) else None

  # For AMDIM, same flipping applied to both copies of image
  amdim_flip = double_sample and tf_name == 'FlipLR'
  if amdim_flip: n_imgs //= 2

  # Preset args so same transformation applied to data/targets
  shuffled_idxs = torch.randperm(n_imgs, device=data.device)
  opts = shuffled(transform.options(data.shape), inplace=True)
  tf_args = [transform, shuffled_idxs, opts[:max_options]]

  # Apply transformation
  imgs, applied_tfs = apply_tf_aux(data, *tf_args)
  if tgts is not None: tgts, _ = apply_tf_aux(targets, *tf_args)

  if amdim_flip:
    # Apply same flip to other copy of images
    tf_args[1] += n_imgs
    imgs_2, _ = apply_tf_aux(data, *tf_args)
    if tgts is not None: tgts_2, _ = apply_tf_aux(targets, *tf_args)

    # Concatenate result
    imgs = torch.cat([imgs, imgs_2])
    applied_tfs = np.concatenate([applied_tfs,applied_tfs])
    if tgts is not None: tgts = torch.cat([tgts, tgts_2])

  return imgs, tgts, applied_tfs


def shuffled(xs, inplace=False):
  xs = xs if inplace else copy.copy(xs)
  np.random.shuffle(xs)
  return xs


def flip_lr(x):
  if isinstance(x, torch.Tensor):
    return torch.flip(x, [-1])
  return x[..., ::-1].copy()


class Crop(namedtuple('Crop', ('h', 'w', 'pad'))):
  def apply(self, x, x0, y0):
    tmp = torch.zeros_like(x)
    src_x0, src_y0 = max(x0 - self.pad, 0), max(y0 - self.pad, 0)
    src_x1, src_y1 = x0 + self.w - self.pad, y0 + self.h - self.pad
    dst_x0, dst_y0 = max(0, self.pad - x0), max(0, self.pad - y0)
    dst_x1, dst_y1 = self.pad - x0 + self.w, self.pad - y0 + self.h
    tmp[..., dst_y0:dst_y1, dst_x0:dst_x1] = x[..., src_y0:src_y1, src_x0:src_x1]
    return tmp

  def options(self, shape):
    *_, H, W = shape
    H += 2 * self.pad
    W += 2 * self.pad
    all_pos = [{'x0': x0, 'y0': y0} for x0 in range(W+1-self.w) for y0 in range(H+1-self.h)]
    no_crop_pos = [{'x0': self.pad, 'y0': self.pad} for _ in range(int(len(all_pos) * .25))]
    return all_pos + no_crop_pos


class FlipLR(namedtuple('FlipLR', ())):
  def apply(self, x, choice):
    return flip_lr(x) if choice else x

  def options(self, shape):
    return [{'choice': b} for b in [True, False]]


class Grayscale(namedtuple('Grayscale', ())):
  def apply(self, x, choice):
    if choice:
      tmp = x.clone()
      tmp[:] = x.mean(1, keepdims=True)
      return tmp
    else:
      return x

  def options(self, shape):
    return [{'choice': b} for b in [True, False, False, False]]


class ColorJitter(namedtuple('ColorJitter', ())):
  def apply(self, x, choice):
    if not choice:
      tmp = x.clone()
      for i in range(3):
        tmp[:, i] *= 2 ** (np.random.randn() * .25)
        tmp[:, i] += np.random.randn() * .25
      return tmp
    else:
      return x

  def options(self, shape):
    return [{'choice': b} for b in [True, False, False, False, False] * 100]


class Cutout(namedtuple('Cutout', ('h', 'w'))):
  def apply(self, x, x0, y0):
    x[..., y0:y0+self.h, x0:x0+self.w] = 0.0
    return x

  def options(self, shape):
    *_, H, W = shape
    return [{'x0': x0, 'y0': y0} for x0 in range(W+1-self.w) for y0 in range(H+1-self.h)]


class Rotate90(namedtuple('Rotate90', ())):
  def apply(self, x, choice):
    if choice == 0:
      return x
    elif choice == 1:                 # Rotate 90
      return torch.flip(x.transpose(-2,-1), [-1])
    elif choice == 2:                 # Rotate 180
      return torch.flip(x, [-2,-1])
    elif choice == 3:                 # Rotate 270
      return torch.flip(x.transpose(-2,-1), [-2])

  def options(self, shape):
    return [{'choice': i} for i in range(4)]
