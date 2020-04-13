import argparse
import gin
import importlib
import subprocess

import ray
from ray import tune

from selfstudy import session, paths
from selfstudy.util import helper

def parse_command_line():
  parser = argparse.ArgumentParser()

  parser.add_argument('-e', '--exp_id', type=str, default='default')
  parser.add_argument('-r', '--ray_config', type=str, default='single_run')
  parser.add_argument('-g', '--gin_config', nargs='+', default=['cifar/r9/classify'],
    help='Set of config files for gin (separated by spaces) '
         'e.g. --gin_config file1 file2 (exclude .gin from path)')
  parser.add_argument('-p', '--gin_param', nargs='+', default=[],
    help='Parameter settings that override config defaults '
         'e.g. --gin_param \'module_1.a = 2\' \'module_2.b = 3\'')

  parser.add_argument('-l', '--local_mode', action='store_true')
  parser.add_argument('-a', '--redis_address', type=str, default=None)
  parser.add_argument('-q', '--quiet', action='store_true')
  parser.add_argument('-c', '--checkpoint_at_end', action='store_true')
  parser.add_argument('--resume', action='store_true')

  return parser.parse_args()


def main():
  flags = parse_command_line()

  # Start Ray
  ray.init(local_mode=flags.local_mode,
           redis_address=flags.redis_address)

  # Load config file with experiment settings
  cfg = importlib.import_module('selfsup.config.' + flags.ray_config)
  cfg.exp_args['config']['flags'] = flags

  # Command line flags
  if flags.quiet: cfg.exp_args['verbose'] = False
  if flags.resume: cfg.exp_args['resume'] = True
  if flags.checkpoint_at_end: cfg.exp_args['checkpoint_at_end'] = True

  # Run experiment
  trials = tune.run(session.SessionManager,
                    name=flags.exp_id,
                    trial_name_creator=helper.ray_trial_str_creator,
                    local_dir=paths.EXP_DIR,
                    **cfg.exp_args)


if __name__ == '__main__':
  main()
