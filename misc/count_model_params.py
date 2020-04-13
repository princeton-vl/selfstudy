import argparse
import gin
from tqdm import tqdm
import time

from selfstudy import session, paths, models

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gin_config', nargs='+', default=['syn/supervised'])
parser.add_argument('-p', '--gin_param', nargs='+', default=[])
flags = parser.parse_args()

gin_files = ['%s/%s.gin' % (paths.CONFIG_DIR, g) for g in flags.gin_config]
gin.parse_config_files_and_bindings(gin_files, flags.gin_param)

# Initialize model
model = 'Wrapper'
m = models.__dict__[model]()

# Print number of parameters
print('Wrapper.model_choice:',gin.query_parameter('Wrapper.model_choice'))
print('init_resnet.choice:', gin.query_parameter('init_resnet.choice'))
print(sum(p.numel() for p in m.parameters()), 'parameters')
