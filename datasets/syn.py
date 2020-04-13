import h5py
import gin
import numpy as np
import torch

from .. import paths
from .base import BaseDataset


class SynDataset(BaseDataset):
  @gin.configurable('SynDataset')
  def get_dataset_path(self, dataset_choice='lg_3222', use_lab=True):
    data_path = f'{paths.DATA_DIR}/syn/{dataset_choice}/data_{self.input_res}'
    if use_lab: data_path += '_lab'

    self.data_path = f'{data_path}.h5'

    return self.data_path

  def initialize_data(self):
    self.get_dataset_path()

    # Get total dataset size
    with h5py.File(self.data_path, 'r') as data:
      n_samples = data['data'].shape[0]

    # Open HDF5 file
    if self.data is None and not self.remote_loading:
      self.data = h5py.File(self.data_path, 'r')

    # Define indices for train/val/test split
    n_train = n_samples * 5 // 6
    test_offset = n_samples // 12
    n_valid = self.num_valid

    base_idxs = {
      'train': np.arange(n_train),
      'valid': np.arange(n_valid) + n_train,
      'test': np.arange(n_valid) + n_train + test_offset,
    }
    self.base_idxs = base_idxs[self.split]
