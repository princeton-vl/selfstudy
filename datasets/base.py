import h5py
import gin
import numpy as np
import ray

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .. import paths
from ..util import helper
from ..util import augment as aug


def load_from_hdf5(k, path, idxs, chunk_size, is_file=False):
  tmp_idxs = idxs[idxs % chunk_size == 0]

  if is_file:
    d = [torch.tensor(path[k][i:i+chunk_size]) for i in tmp_idxs]
  else:
    with h5py.File(path, 'r') as f:
      d = [torch.tensor(f[k][i:i+chunk_size]) for i in tmp_idxs]

  return torch.cat(d, 0)


def load_data(path, idxs, chunk_size, predict_pose, is_file=False):
  d = load_from_hdf5('data', path, idxs, chunk_size, is_file)
  t = load_from_hdf5('targets', path, idxs, chunk_size, is_file)

  if predict_pose:
    r = load_from_hdf5('rot_mat', path, idxs, chunk_size, is_file)
    gt_pose = helper.get_pose_class_idx(r)
    t = torch.stack([t, gt_pose], 1)

  return d, t


@ray.remote
def load_data_remote(path, idxs, chunk_size, predict_pose):
  return load_data(path, idxs, chunk_size, predict_pose)


@gin.configurable(blacklist=['split', 'preloaded_data'])
class BaseDataset(Dataset):
  def __init__(self, split, input_res=32, num_valid=5000, semisupervised=None,
               double_sample=False, predict_pose=False, max_images=50000,
               chunk_size=1000, use_fp16=True, remote_loading=False,
               preloaded_data=None):
    # Dataset settings
    self.split = split
    self.input_res = input_res
    self.num_valid = num_valid
    self.semisupervised = semisupervised
    self.double_sample = double_sample
    self.predict_pose = predict_pose

    # Memory settings
    self.max_images = max_images
    if self.double_sample: self.max_images //= 2
    self.chunk_size = chunk_size
    self.use_fp16 = use_fp16
    fp_type = torch.float16 if use_fp16 else torch.float32
    self.on_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dataloading options
    self.data = preloaded_data
    self.remote_loading = remote_loading

    # Initialize everything else
    self.initialize_data()
    self.setup_idxs()
    self.setup_aug()
    self.apply_transforms()

  def initialize_data(self):
    raise NotImplementedError('initialize_data() must be implemented.')

  def setup_idxs(self, min_len=10000):
    # Figure out subset of indices to use during semisupervised training
    self.idxs = self.base_idxs
    if self.split == 'train' and self.semisupervised:
      self.idxs = self.idxs[:self.semisupervised]
      if len(self.idxs) == 4000:
        min_len = 20000
      if len(self.idxs) < min_len:
        to_repeat = int(np.ceil(min_len / len(self.idxs)))
        self.idxs = np.concatenate([self.idxs] * to_repeat)

    self.ref_idxs = self.idxs.copy()

    # Number of subsets needed to run through all data given the max_image limit
    # A true epoch is after 'num_divisions' rounds of training has passed
    self.total_samples = len(self.idxs)
    self.max_images = min(self.total_samples, self.max_images)
    self.chunk_size = min(self.chunk_size, self.max_images)
    if self.split == 'train' and self.semisupervised is not None:
      self.chunk_size = min(self.chunk_size, self.semisupervised)

    try:
      check_val = gin.query_parameter('CMCModel.dataset_size')
      with gin.unlock_config():
        print('Setting CMC memory size to', self.total_samples)
        gin.bind_parameter('CMCModel.dataset_size', self.total_samples)
    except:
      pass

    self.cached_data = None
    self.cached_idxs = None
    self.curr_data = None
    self.d_idx = -1

    self.num_divisions = int(np.ceil(self.total_samples / self.max_images))
    print(f'Epoch is divided into {self.num_divisions} rounds.')

    if self.num_divisions > 1:
      self.preload_data()

  @gin.configurable('augment')
  def setup_aug(self, do_augment=None, color_jitter=True, grayscale=True,
                flip=True, crop=True, cutout=True, transform_targets=False):
    res = self.input_res

    if do_augment is None: do_augment = self.split == 'train'
    self.augment = do_augment
    self.transform_targets = transform_targets
    self.transforms = []

    if do_augment:
      if color_jitter:
        self.transforms += [aug.ColorJitter()]
      if grayscale:
        self.transforms += [aug.Grayscale()]
      if flip:
        self.transforms += [aug.FlipLR()]
      if crop:
        self.transforms += [aug.Crop(res, res, res // 8)]
      if cutout:
        self.transforms += [aug.Cutout(res // 4, res // 4)]

  def collect_data(self, idxs):
    if self.remote_loading:
      return load_data_remote.remote(self.data_path, idxs, self.chunk_size,
                                     self.predict_pose)
    else:
      return load_data(self.data, idxs, self.chunk_size,
                       self.predict_pose, is_file=True)

  def move_to_device(self, data, device=None):
    """Ensure data is in proper format on correct device."""
    if device is None: device = self.on_device

    if self.remote_loading:
      d, t = helper.ray_get_and_free(data)
    else:
      d, t = data

    if not self.use_fp16: d = d.float()
    return d.to(device), t.to(device)

  def preload_data(self):
    """Preload data for next division.

    Only happens in parallel if `remote_loading = True`.
    """
    self.d_idx = (self.d_idx + 1) % self.num_divisions

    if self.d_idx == 0:
      # Start of new epoch, shuffle indices to mix up samples across divisions
      self.ref_idxs = self.ref_idxs.reshape(-1, self.chunk_size)
      self.ref_idxs = np.random.permutation(self.ref_idxs).flatten()

    i0 = self.d_idx * self.max_images
    tmp_idxs = self.ref_idxs[i0:i0+self.max_images]
    self.cached_data = self.collect_data(tmp_idxs)
    self.cached_idxs = tmp_idxs.copy()

  def reset_data(self):
    """Reset data augmentation, load next batch of data.

    If the entire dataset does not fit on the GPU we process it separate sets,
    this function will load in the next division of the dataset.
    """
    self.applied_tfs = []

    if self.num_divisions == 1:
      if self.cached_data is None:
        # Full dataset fits on the GPU, cache an un-augmented version
        data = self.collect_data(self.ref_idxs)
        self.cached_data = self.move_to_device(data)

      self.curr_data, self.curr_tgts = self.cached_data
      self.curr_idxs = self.ref_idxs

    else:
      self.curr_data, self.curr_tgts = self.move_to_device(self.cached_data)
      self.curr_idxs = self.cached_idxs
      self.preload_data()

    if self.double_sample:
      # Concatenate two copies of data together
      self.curr_data = torch.cat([self.curr_data, self.curr_data])
      self.curr_tgts = torch.cat([self.curr_tgts, self.curr_tgts])

  def apply_transforms(self, transforms=None, reset=True, max_options=500):
    """Apply all data augmentation to data and targets."""
    if transforms is None: transforms = self.transforms
    if reset: self.reset_data()
    for t in transforms:
      imgs, tgts, tfs = aug.apply_tf(self.curr_data, t, self.curr_tgts,
                                     self.transform_targets, self.double_sample,
                                     max_options=max_options)
      self.curr_data = imgs
      if tgts is not None: self.curr_tgts = tgts
      self.applied_tfs += [tfs]

  def __getitem__(self, idx):
    """Get data sample including image, target, and sample index.

    If double sample is True, then a concatenated (2 x 3 x res x res) image will
    be returned with two different augmented versions of the same image.

    The sample_idx returns the actual index of the sample from the full dataset
    and not just within the current data block. This is important for indexing
    into memory during CMC training.
    """
    sample_idx = self.curr_idxs[idx]
    img, label = self.curr_data[idx], self.curr_tgts[idx]

    if self.double_sample:
      img = torch.stack([img, self.curr_data[idx + len(self)]])

    return img, label, sample_idx

  def __len__(self):
    """Return number of available samples in current data block.

    Note: This is not the total dataset size, just the data currently available
    on the GPU.
    """
    l = len(self.curr_data)
    if self.double_sample: l //= 2

    return l
