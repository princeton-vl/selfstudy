import argparse
import gin
import time
import torch
from tqdm import tqdm
import numpy as np

from torch.functional import F

from selfstudy import session, paths
from selfstudy.util import helper

from sklearn.linear_model import LogisticRegression, LinearRegression


def get_and_save_feats(exp_id=None, dense=False, include_test=False):
  """Load a pretrained model and run through dataset to get output features."""

  with gin.unlock_config():
    gin.bind_parameter('augment.do_augment', False)
    if include_test:
      gin.bind_parameter('session.test_iters', -1)

  # Initialize session
  sess = session.initialize_session()
  model = sess['model']
  loader = sess['loader']
  iters = sess['iters']
  task = sess['task']

  if exp_id is None:
    exp_id = sess['restore_session']

  # Restore model
  if exp_id is not None:
    path = '%s/%s/snapshot' % (paths.EXP_DIR, exp_id)
    print("Restoring from:", path)
    if not torch.cuda.is_available():
      loaded = torch.load(path, map_location='cpu')
    else:
      loaded = torch.load(path)

    model.load_state_dict(loaded['model'], strict=False)

  model.cuda()
  model.eval()

  n_feats = model.backbone.out_feats
  print("# output features", n_feats)

  splits = list(loader.datasets.keys())
  results = {s:{} for s in splits}
  idx_offsets = {s:loader.datasets[s].idxs[0] for s in splits}

  if not dense:
    all_feats = {s:np.zeros((len(loader.datasets[s]), n_feats)) for s in splits}
    all_labels = {s:np.zeros(len(loader.datasets[s])) for s in splits}
  else:
    all_feats, all_labels = None, None

  for split in splits:
    for _ in tqdm(range(iters[split])):
      # Load sample + save reference labels
      sample = loader.get_sample(split)
      idxs = np.array(sample[2]) - idx_offsets[split]

      if not dense:
        labels = np.array(sample[1].cpu())
        all_labels[split][idxs] = labels
      else:
        labels = F.interpolate(sample[1].float(), size=r)
        all_labels[split][idxs] = np.array(labels.cpu())

      if torch.cuda.is_available():
        sample = [s.cuda() for s in sample]

      # Pass through model
      with torch.no_grad():
        model_out = task.forward(model, sample)
        loss, eval_metrics = task.evaluate(sample, model_out)

      # Save features
      feats = model_out[2]
      all_feats[split][idxs] = np.array(feats.cpu())

      # Collect results
      to_report = {'loss': loss.item()}
      for k,v in eval_metrics.items():
        to_report[k] = v.item()

      for k,v in to_report.items():
        if not k in results[split]:
          results[split][k] = []
        results[split][k] += [v]

    # Print out results (sanity check that loaded model is good)
    for k in results[split]:
      print(k, np.array(results[split][k]).mean())


  if exp_id is not None:
    # Save features + labels
    torch.save({'train_feats': all_feats['train'].astype(np.float16),
                'valid_feats': all_feats['valid'].astype(np.float16),
                'train_labels': all_labels['train'].astype(np.uint8),
                'valid_labels': all_labels['valid'].astype(np.uint8),
                'idx_offsets': idx_offsets,},
               '%s/%s/network_output.pt' % (paths.EXP_DIR, exp_id))

  return all_feats, all_labels


def train_lr_weights(exp_id, feats, labels, data_amts=None, dense=False, is_depth=False):
  """Get weights of final FC layer conditioned only on fraction of dataset."""
  if data_amts is None:
    if not dense:
      data_amts = [100, 250, 1000, 4000, 16000, 50000]
    else:
      data_amts = [100, 250, 1000, 4000]

  for i in data_amts:
    # Get indices to training data
    idxs = helper.get_train_val_idxs('train', labels['train'], 0, 10,
                                     semisupervised=i, min_len=0)

    if dense:
      # Reshape/subsample/flatten feats
      tmp_f = feats['train'].transpose(0,2,3,1)
      tmp_f = tmp_f[idxs].reshape(-1, tmp_f.shape[-1])
      tmp_l = labels['train'][idxs,1].reshape(-1)

      subsample = np.random.permutation(np.arange(len(tmp_l)))[:50000]
      tmp_f = tmp_f[subsample]
      tmp_l = tmp_l[subsample]

      tmp_f_valid = feats['valid'].transpose(0,2,3,1)
      tmp_f_valid = tmp_f_valid.reshape(-1, tmp_f_valid.shape[-1])
      tmp_l_valid = labels['valid'][:,1].reshape(-1)

      subsample = np.random.permutation(np.arange(len(tmp_l_valid)))[:5000]
      tmp_f_valid = tmp_f_valid[subsample]
      tmp_l_valid = tmp_l_valid[subsample]

    else:
      tmp_f = feats['train'][idxs]
      tmp_l = labels['train'][idxs]

      tmp_f_valid = feats['valid']
      tmp_l_valid = labels['valid']

    # Train logistic regression model using data subset
    lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=500)
    lr.fit(tmp_f, tmp_l)

    # Get predictions and evaluate on the full training and validation data
    pred = lr.predict(tmp_f)
    t_acc = np.array(pred == tmp_l).mean()
    pred = lr.predict(tmp_f_valid)
    v_acc = np.array(pred == tmp_l_valid).mean()

    print(i, 'train:', t_acc, 'valid:', v_acc)

    # Save a reference torch linear layer to load later
    fc = torch.nn.Linear(feats['train'].shape[1], 10)
    fc.weight[:] = torch.Tensor(lr.coef_)
    fc.bias[:] = torch.Tensor(lr.intercept_)

    torch.save({'model': fc.state_dict(),
                'train_accuracy': t_acc,
                'valid_accuracy': v_acc},
               f'{paths.EXP_DIR}/{exp_id}/fc_{i}.pt')


def main():
  # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--exp_id', type=str, default=None)
  parser.add_argument('-g', '--gin_config', nargs='+', default=[])
  parser.add_argument('-p', '--gin_param', nargs='+', default=[])
  parser.add_argument('-d', '--data_amts', type=int, nargs='+', default=None)
  parser.add_argument('--get_feats', action='store_true')
  parser.add_argument('--train_lr', action='store_true')
  parser.add_argument('--dense', action='store_true')
  parser.add_argument('--include_test', action='store_true')
  flags = parser.parse_args()

  gin_files = ['%s/%s.gin' % (paths.CONFIG_DIR, g) for g in flags.gin_config]
  gin.parse_config_files_and_bindings(gin_files, flags.gin_param)

  if flags.get_feats:
    print('Generating features...')
    feats, labels = get_and_save_feats(flags.exp_id, dense=flags.dense, include_test=flags.include_test)
  else:
    print('Loading features...')
    r = torch.load('%s/%s/network_output.pt' % (paths.EXP_DIR, flags.exp_id))
    feats = {s:r[f'{s}_feats'] for s in ['train', 'valid']}
    labels = {s:r[f'{s}_labels'] for s in ['train', 'valid']}

  if flags.train_lr:
    print('Training logistic regression weights...')
    train_lr_weights(flags.exp_id, feats, labels, flags.data_amts, dense=flags.dense)


if __name__ == '__main__':
  main()
