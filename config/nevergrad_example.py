import nevergrad as ng
import numpy as np
from ray.tune import schedulers
from ray.tune.suggest.nevergrad import NevergradSearch

budget = 400

opt_args = {
  'scheduler.target_lrs': ng.var.Log(1e-4, 1e-2),
  'scheduler.final_lr': ng.var.Log(1e-7, 1e-4),
  'optimizer.weight_decay': ng.var.Log(1e-7, 1e-3),
  'scheduler.exp_factor': ng.var.Scalar().bounded(-.5, 2),
}

inst = ng.Instrumentation(**opt_args)
opt = ng.optimizers.registry['CMA'](instrumentation=inst, budget=budget)
search_alg = NevergradSearch(opt, None, max_concurrent=4, reward_attr='valid_accuracy', mode='max')

exp_args = {
  'resume': False,
  'search_alg': search_alg,
  'scheduler': schedulers.FIFOScheduler(),
  'stop': {'is_finished': True},
  'verbose': True,
  'num_samples': budget,
  'checkpoint_at_end': False,
  'resources_per_trial': {'cpu': 2, 'gpu': 1},
  'config': {}
}
