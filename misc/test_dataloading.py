import argparse
import gin
from tqdm import tqdm
import time
import ray

from selfstudy import session, paths, datasets
from selfstudy.util import data

# Config settings to load appropriate dataset
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gin_config', nargs='+', default=['test_dataloader'])
parser.add_argument('-p', '--gin_param', nargs='+', default=[])
flags = parser.parse_args()

gin_files = ['%s/%s.gin' % (paths.CONFIG_DIR, g) for g in flags.gin_config]
gin.parse_config_files_and_bindings(gin_files, flags.gin_param)

# Start ray if testing remote loading
remote_loading = gin.query_parameter('BaseDataset.remote_loading')
if remote_loading:
  ray.init()

# Initialize dataset
ds_choice = gin.query_parameter('session.dataset')
loader = data.LocalDataloader(ds_choice, train_iters=-1, valid_iters=0)

# Load several epochs
for epoch_idx in range(5):
  print(epoch_idx)
  s_time = time.time()
  for i in tqdm(range(loader.iters['train'])):
    sample = loader.get_sample('train')
  print('total time: %.2f' % (time.time() - s_time))
