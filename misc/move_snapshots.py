import argparse
import json
import os
import subprocess

from selfstudy import paths
from selfstudy.util import helper

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--exp_id', type=str, default=None)
parser.add_argument('-m', '--move_files', action='store_true')
parser.add_argument('-p', '--param_str', type=str, default='')
flags = parser.parse_args()

exp_dir = f'{paths.EXP_DIR}/{flags.exp_id}'
exp_states = []

def ckpt_path_mapping(cfg):
  return flags.param_str.format(cfg=cfg)

try:
  exps = os.listdir(exp_dir)
  for e in exps:
    if 'experiment_state' in e:
      exp_states += [e]
except FileNotFoundError:
  print(f'Experiment {exp_dir} does not exist')
  exit()

assert len(exp_states) != 0, f'No experiment state found. ({exp_dir})'

for exp_state in exp_states:
  # Read in experiment state
  with open(f'{exp_dir}/{exp_state}','r') as f:
    r = json.load(f)

  for e in r['checkpoints']:
    cfg = e['config']
    ckpt_path = e['logdir']
    ckpt_path = f'{exp_dir}/{ckpt_path.split("/")[-1]}'
    ckpt_files = os.listdir(ckpt_path)
    ckpt_dir = None
    max_num = -1

    for f in ckpt_files:
      if 'checkpoint' in f:
        n = int(f.split('_')[-1])
        if n > max_num:
          ckpt_dir = f
          max_num = n

    if ckpt_dir is None:
      print('No checkpoint found in:', ckpt_path)
    else:
      ckpt_path += f'/{ckpt_dir}'
      dest_dir = f'{exp_dir}/{ckpt_path_mapping(cfg)}'
      print('Moving from:', ckpt_path, 'to:', dest_dir)

      if flags.move_files:
        helper.mkdir_p(dest_dir)
        subprocess.call(['mv', ckpt_path + '/snapshot', dest_dir + '/snapshot'])
