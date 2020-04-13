import numpy as np
from ray import tune

exp_args = {
  'stop': {'is_finished': True},
  'resume': False,
  'verbose': True,
  'checkpoint_freq': 0,
  'checkpoint_at_end': False,
  'num_samples': 1,
  'resources_per_trial': {'cpu': 2, 'gpu': 1},
  'config': {
    'SynDataset.dataset_choice': tune.grid_search(
        ['0000','0100','0110','0001','0101','0010',
         '0111','1000','1011','1100','1110','1111']),
    'BaseDataset.semisupervised': tune.grid_search([100, 250, 1000, 4000, 20000,
                                                    50000, 100000, 200000]),
    'scheduler.milestones': tune.sample_from(lambda spec: [5,200] if spec.config['BaseDataset.semisupervised'] < 20000 else [5,600]),
    'Wrapper.pretrained': tune.sample_from(lambda spec: 'pretrained-amdim/' + spec.config['SynDataset.dataset_choice']),
  }
}
