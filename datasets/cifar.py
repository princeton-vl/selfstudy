import gin
import h5py
import numpy as np
import os
from PIL import Image
from skimage import color
import torch
from torchvision import datasets
from tqdm import tqdm

from .. import paths
from ..util import helper
from .base import BaseDataset


cifar_dir = f'{paths.DATA_DIR}/cifar'


def preprocess_cifar_data(to_lab=True, use_100=False):
  """One-time preprocessing of CIFAR data."""
  res = 32
  ds_suffix = ''
  normalize_vals = np.array([[125.32, 122.97, 113.89],
                             [62.99,  62.09,  66.70]])

  if use_100:
    d = datasets.CIFAR100
    ds_suffix += '100'
  else:
    d = datasets.CIFAR10
    ds_suffix += '10'

  if to_lab:
    normalize_vals = np.array([[50.84,  0.40,  5.73],
                               [24.26, 10.15, 16.08]])
    ds_suffix += '_lab'

  # Load raw data
  d_train = d(paths.DATA_DIR, train=True, download=True)
  d_test = d(paths.DATA_DIR, train=False, download=True)

  imgs = np.concatenate([d_train.data, d_test.data])
  targets = np.concatenate([d_train.targets, d_test.targets]).astype(int)

  n_images = len(imgs)
  data = np.zeros((n_images, res, res, 3), dtype=np.float32)
  data_path = f'{cifar_dir}/data_{res}_{ds_suffix}.h5'

  for i in tqdm(range(n_images)):
    tmp_im = imgs[i]

    # Resize image
    tmp_im = Image.fromarray(tmp_im[:,:,:3])
    if res != 32: tmp_im = tmp_im.resize([res, res])

    if to_lab:
      # Convert to lab color space
      tmp_im = color.rgb2lab(tmp_im)

    data[i] = np.array(tmp_im)

  # Preprocess all images
  data = (data - normalize_vals[0]) / normalize_vals[1]
  data = data.transpose(0,3,1,2)
  data = data.astype(np.float16)

  # Save data
  print(f'Saving to: {data_path}')
  helper.mkdir_p(cifar_dir)
  with h5py.File(data_path, 'w') as f:
    f['data'] = data
    f['targets'] = targets


class CIFARDataset(BaseDataset):
  @gin.configurable('CIFARDataset')
  def get_dataset_path(self, to_lab=True, use_100=False):
    ds_suffix = '100' if use_100 else '10'
    if to_lab: ds_suffix += '_lab'

    data_path = f'{cifar_dir}/data_32_{ds_suffix}.h5'

    # Check that it exists, if not, create it
    if not os.path.exists(data_path):
      print(f'Running one-time preprocessing for CIFAR{ds_suffix}.')
      preprocess_cifar_data(to_lab, use_100)

    self.data_path = data_path
    return data_path

  def initialize_data(self):
    # Load CIFAR10/CIFAR100
    self.get_dataset_path()

    if self.data is None and not self.remote_loading:
      self.data = h5py.File(self.data_path, 'r')

    n_train = 45000
    n_valid = 5000
    n_test = 10000

    base_idxs = {
      'train': np.arange(n_train),
      'valid': np.arange(n_valid) + n_train,
      'test': np.arange(n_test) + n_train + n_valid,
    }
    self.base_idxs = base_idxs[self.split]
