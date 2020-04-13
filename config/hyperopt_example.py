import numpy as np
from ray.tune import schedulers
from hyperopt import hp
from ray.tune.suggest.hyperopt import HyperOptSearch

scheduler = schedulers.AsyncHyperBandScheduler(time_attr='training_iteration',
                                               metric='valid_accuracy', mode='max',
                                               max_t=100, grace_period=20)

space = {
  'scheduler.target_lrs': hp.loguniform('scheduler.target_lrs', np.log(1e-4), np.log(1e-2)),
  'scheduler.final_lr': hp.loguniform('scheduler.final_lr', np.log(1e-7), np.log(1e-4)),
  'optimizer.weight_decay': hp.loguniform('optimizer.weight_decay', np.log(1e-7), np.log(1e-3)),
  'scheduler.exp_factor': hp.uniform('scheduler.exp_factor', -.5, 2),
}

search_alg = HyperOptSearch(space, max_concurrent=4, reward_attr='valid_accuracy')

exp_args = {
  'resume': False,
  'search_alg': search_alg,
  'scheduler': scheduler,
  'verbose': True,
  'num_samples': 1024,
  'checkpoint_at_end': False,
  'resources_per_trial': {'cpu': 2, 'gpu': 1},
  'config': {}
}
