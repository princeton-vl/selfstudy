import gin
import ray
import time
import numpy as np

from torch.utils.data import DataLoader

from .. import datasets


@gin.configurable('dataloader', blacklist=['dataset'])
def initialize_dataloader(dataset, batch_size=8, shuffle=True, num_workers=0):
  return DataLoader(dataset, batch_size=batch_size,
                    shuffle=shuffle, num_workers=num_workers)


class LocalDataloader():
  def __init__(self,
               dataset_name,
               train_iters=-1,
               valid_iters=-1,
               test_iters=-1):

    self.loaders = {}
    self.count = 0
    self.iters = {'train': train_iters,
                  'valid': valid_iters,
                  'test': test_iters}

    self.datasets = {}
    preloaded_data = None

    splits = [s for s in self.iters if self.iters[s] != 0]

    for split in splits:
      d = datasets.__dict__[dataset_name](split, preloaded_data=preloaded_data)
      self.datasets[split] = d

      if preloaded_data is None and not d.remote_loading:
        preloaded_data = d.data

      with gin.config_scope(split):
        self.loaders[split] = initialize_dataloader(d)
      if self.iters[split] == -1:
        self.iters[split] = len(self.loaders[split])

    self.data_iterator = {s: iter(self.loaders[s]) for s in self.loaders}
    self.cached_samples = {}

  def get_sample(self, split):
    try:
      s = self.data_iterator[split].next()
    except StopIteration:
      self.datasets[split].apply_transforms()
      self.data_iterator[split] = iter(self.loaders[split])
      s = self.data_iterator[split].next()
    return s

  def get_iters(self):
    return self.iters

  def apply_transforms(self, split):
    self.datasets[split].apply_transforms()

  def use_fp16(self):
    if 'use_fp16' in self.datasets['train'].__dict__:
      return self.datasets['train'].use_fp16
    else:
      return False
