import gin
import importlib
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.optim

import apex.optimizers
from apex import amp
from ray import tune

from . import paths, models, tasks
from .util import data, helper


@gin.configurable('optimizer', blacklist=['params'])
def initialize_optimizer(params, optim_fn='SGD', lr=1, **kargs):
  if optim_fn == 'FusedAdam':
    return apex.optimizers.FusedAdam(params, lr=lr, **kargs)
  else:
    return torch.optim.__dict__[optim_fn](params, lr=lr, **kargs)


@gin.configurable('scheduler', blacklist=['optimizer'])
def initialize_scheduler(optimizer, iters, metric='valid_loss',
                         scheduler_fn='LambdaLR', **kargs):
  last_epoch = kargs['milestones'][-1]
  if scheduler_fn == 'LambdaLR':
    # Add default values for specific kargs
    if 'exp_factor' not in kargs:
      kargs['exp_factor'] = 0

    lr_vals = helper.define_lr_vals(kargs['milestones'],
                                    kargs['target_lrs'],
                                    max(1, iters['train']),
                                    exp_factor=kargs['exp_factor'])
    lr_lambda = lambda x: lr_vals[x] if x < len(lr_vals) else lr_vals[-1]
    kargs = {'lr_lambda': lr_lambda}

  elif scheduler_fn == 'ReduceLROnPlateau':
    del kargs['milestones']

  scheduler = torch.optim.lr_scheduler.__dict__[scheduler_fn](optimizer, **kargs)

  return scheduler, scheduler_fn, metric, last_epoch


@gin.configurable('session')
def initialize_session(
  model='Wrapper',
  dataset='SynDataset',
  task='ClassifyTask',
  train_iters=100,
  valid_iters=10,
  test_iters=0,
  use_apex=True,
  keep_bn_f32=True,
  restore_session=None):

  # Dataset
  print("Loading dataset...")
  dataloader_args = []
  loader = data.LocalDataloader(dataset, train_iters, valid_iters, test_iters)

  # Model
  print("Initializing model...")
  m = models.__dict__[model]()
  if torch.cuda.is_available(): m.cuda()
  if not use_apex and loader.use_fp16(): m.half()

  optimizer = initialize_optimizer(m.parameters())
  if use_apex:
    m, optimizer = amp.initialize(m, optimizer, opt_level='O2',
                                  keep_batchnorm_fp32=keep_bn_f32)

  tmp_optimizer = optimizer
  if not keep_bn_f32:
    tmp_optimizer = optimizer.optimizer

  scheduler, scheduler_fn, schedule_metric, last_epoch = initialize_scheduler(tmp_optimizer, loader.iters)

  # Task
  t = tasks.__dict__[task]()

  return {
    'task': t,
    'model': m,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'schedule_metric': schedule_metric,
    'scheduler_fn': scheduler_fn,
    'last_epoch': last_epoch,
    'loader': loader,
    'iters': loader.iters,
    'restore_session': restore_session,
    'use_apex': use_apex,
  }


class SessionManager(tune.Trainable):
  def _setup(self, ray_config):
    """Initialize training session.

    Load and configure relevant dataset and model, and handle restoring from
    checkpoint if necessary.
    """

    self.ray_config = ray_config
    self.step = {'train': 0, 'valid': 0, 'test': 0}
    self.log = {}
    self.valid_accuracy = []
    self.valid_loss = []
    self.epoch_count = 0
    self.best_accuracy = 0

    # Parse config files and command line args
    gin_files = [f'{paths.CONFIG_DIR}/{g}.gin' for g in ray_config['flags'].gin_config]
    gin.parse_config_files_and_bindings(gin_files, ray_config['flags'].gin_param)

    # Update config according to ray experiment
    with gin.unlock_config():
      for k in ray_config:
        if k not in ['flags', 'loader', 'misc']:
          print("Setting %s to" % k, ray_config[k])
          gin.bind_parameter(k, ray_config[k])

    if ray_config['flags'].quiet:
      helper.suppress_output()

    # Setup model, dataset, and task
    s = initialize_session()
    for k in s: self.__dict__[k] = s[k]

    # Define what to save/load in model checkpoints
    self.checkpoint_ref = ['model', 'optimizer', 'scheduler', 'log', 'step']

    # Check whether to restore from an old checkpoint
    if self.restore_session is not None:
      print("Loading previous checkpoint: %s" % self.restore_session)
      self._restore('%s/%s/snapshot' % (paths.EXP_DIR, self.restore_session))

    # Print out current experiment config
    print(gin.operative_config_str())

  def _train(self):
    splits = ['train', 'valid', 'test']
    self.epoch_count += 1

    for s in splits:
      # Track accuracy/loss vals
      self.total_accuracy = []
      self.total_loss = []

      # Set network mode
      if s == 'train': self.model.train()
      else: self.model.eval()

      # Run network for appropriate number of steps
      for _ in range(self.iters[s]):
        self.run(self.step[s], s)
        self.step[s] += 1

      # Average accuracy/loss across entire round
      if s != 'train' and self.iters[s] > 0:
        self.log[f'{s}_accuracy'] = np.array(self.total_accuracy).mean()
        self.log[f'{s}_loss'] = np.array(self.total_loss).mean()

    to_report = 'valid' if self.iters['valid'] else 'train'
    self.log['mean_accuracy'] = self.log[f'{to_report}_accuracy']

    self.best_accuracy = max(self.best_accuracy, self.log['mean_accuracy'])
    self.log['best_accuracy'] = self.best_accuracy

    # Manage LR schedule
    if self.iters['train'] > 0:
      if self.scheduler_fn != 'LambdaLR':
        self.scheduler.step(self.log[self.schedule_metric])

    # Display results
    for s in splits:
      out_str = ''
      for k in self.log:
        if s in k and ('loss' in k or 'acc' in k) and not ('mlp' in k or 'fc' in k):
          tmp_k = ' '.join(k.split('_')[1:]).replace('accuracy', 'acc')
          out_str += f', {tmp_k}: {self.log[k]:.3g}'
      print(s + out_str)
    print(f'Best accuracy: {self.best_accuracy:.3g}')

    self.log['is_finished'] = self.epoch_count >= self.last_epoch

    return self.log

  def run(self, step, split='train'):
    """Sample minibatch and run forward and backward step through model."""

    sample = self.loader.get_sample(split)
    # Move to GPU
    if torch.cuda.is_available():
      sample = [s.cuda() if isinstance(s, torch.Tensor) else s for s in sample]

    # Run forward pass and calculate loss
    model_out = self.task.forward(self.model, sample)
    loss, eval_metrics = self.task.evaluate(sample, model_out)

    # Run backward pass
    if split == 'train':
      self.optimizer.zero_grad()

      if self.use_apex:
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
          scaled_loss.backward()
      else:
        loss.backward()

      self.optimizer.step()

      if self.scheduler_fn == 'LambdaLR':
        self.scheduler.step()

    # Collect and update results
    batch_size = sample[0].shape[0]
    self.total_loss += [loss.item()] * batch_size
    self.total_accuracy += [eval_metrics['accuracy'].item()] * batch_size

    to_report = {'loss': loss.item()}
    for k,v in eval_metrics.items():
      to_report[k] = v.item()

    for k in to_report:
      helper.running_mean(self.log, '%s_%s' % (split, k), to_report[k])

  def checkpoint(self, path, checkpoint_ref=None, action='save'):
    """Load or save intermediate training snapshots."""

    if checkpoint_ref is None: checkpoint_ref = self.checkpoint_ref

    if action == 'save':
      to_save = {l: self.__dict__[l] for l in checkpoint_ref}
      for l, v in to_save.items():
        if hasattr(v, 'state_dict'):
          to_save[l] = v.state_dict()

      torch.save(to_save, path)

    if action == 'load':
      if not torch.cuda.is_available():
        loaded = torch.load(path, map_location='cpu')
      else:
        loaded = torch.load(path)

      for l in checkpoint_ref:
        if l in loaded:
          v = loaded[l]
          if hasattr(self.__dict__[l], 'state_dict'):
            kargs = {}
            if l == 'model': kargs = {'strict': False}
            self.__dict__[l].load_state_dict(v, **kargs)
          else:
            self.__dict__[l] = v

  def _save(self, checkpoint_dir):
    # Make checkpoint directory (if it doesn't exist already)
    helper.mkdir_p(checkpoint_dir)

    # Define snapshot path and save checkpoint
    checkpoint_path = checkpoint_dir + '/snapshot'
    self.checkpoint(checkpoint_path, action='save')

    return checkpoint_path

  def _restore(self, checkpoint_path):
    # Note: different argument usage between _save and _restore (required by ray)
    self.checkpoint(checkpoint_path, action='load')
