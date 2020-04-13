# From: https://github.com/Philip-Bachman/amdim-public.git
# Modified to remove multi-GPU stuff

import gin
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .classify import ClassifyTask


def tanh_clip(x, clip_val=10.):
  '''
  soft clip values to the range [-clip_val, +clip_val]
  '''
  if clip_val is not None:
    x_clip = clip_val * torch.tanh((1. / clip_val) * x)
  else:
    x_clip = x
  return x_clip


def loss_xent(logits, labels, ignore_index=-1):
  '''
  compute multinomial cross-entropy, for e.g. training a classifier.
  '''
  xent = F.cross_entropy(tanh_clip(logits, 10.), labels,
               ignore_index=ignore_index)
  lgt_reg = 1e-3 * (logits**2.).mean()
  return xent + lgt_reg


class NCE_MI_MULTI(nn.Module):
  def __init__(self, tclip=20.):
    super(NCE_MI_MULTI, self).__init__()
    self.tclip = tclip

  def _model_scores(self, r_src, r_trg, mask_mat):
    '''
    Compute the NCE scores for predicting r_src->r_trg.

    Input:
      r_src    : (n_batch_gpu, n_rkhs)
      r_trg    : (n_rkhs, n_batch * n_locs)
      mask_mat : (n_batch_gpu, n_batch)
    Output:
      raw_scores : (n_batch_gpu, n_locs)
      nce_scores : (n_batch_gpu, n_locs)
      lgt_reg    : scalar
    '''
    n_batch_gpu = mask_mat.size(0)
    n_batch = mask_mat.size(1)
    n_locs = r_trg.size(1) // n_batch
    n_rkhs = r_src.size(1)
    # reshape mask_mat for ease-of-use
    mask_pos = mask_mat.unsqueeze(dim=2).expand(-1, -1, n_locs).float()
    mask_neg = 1. - mask_pos

    # compute src->trg raw scores for batch on this gpu
    raw_scores = torch.mm(r_src, r_trg).float()
    raw_scores = raw_scores.reshape(n_batch_gpu, n_batch, n_locs)
    raw_scores = raw_scores / n_rkhs**0.5
    lgt_reg = 5e-2 * (raw_scores**2.).mean()
    raw_scores = tanh_clip(raw_scores, clip_val=self.tclip)

    '''
    pos_scores includes scores for all the positive samples
    neg_scores includes scores for all the negative samples, with
    scores for positive samples set to the min score (-self.tclip here)
    '''
    # (n_batch_gpu, n_locs)
    pos_scores = (mask_pos * raw_scores).sum(dim=1)
    # (n_batch_gpu, n_batch, n_locs)
    neg_scores = (mask_neg * raw_scores) - (self.tclip * mask_pos)
    # (n_batch_gpu, n_batch * n_locs)
    neg_scores = neg_scores.reshape(n_batch_gpu, -1)
    # (n_batch_gpu, n_batch * n_locs)
    mask_neg = mask_neg.reshape(n_batch_gpu, -1)
    '''
    for each set of positive examples P_i, compute the max over scores
    for the set of negative samples N_i that are shared across P_i
    '''
    # (n_batch_gpu, 1)
    neg_maxes = torch.max(neg_scores, dim=1, keepdim=True)[0]
    '''
    compute a "partial, safe sum exp" over each negative sample set N_i,
    to broadcast across the positive samples in P_i which share N_i
    -- size will be (n_batch_gpu, 1)
    '''
    neg_sumexp = \
      (mask_neg * torch.exp(neg_scores - neg_maxes)).sum(dim=1, keepdim=True)
    '''
    use broadcasting of neg_sumexp across the scores in P_i, to compute
    the log-sum-exps for the denominators in the NCE log-softmaxes
    -- size will be (n_batch_gpu, n_locs)
    '''
    all_logsumexp = torch.log(torch.exp(pos_scores - neg_maxes) + neg_sumexp)
    # compute numerators for the NCE log-softmaxes
    pos_shiftexp = pos_scores - neg_maxes
    # compute the final log-softmax scores for NCE...
    nce_scores = pos_shiftexp - all_logsumexp
    return nce_scores, pos_scores, lgt_reg

  def _loss_g2l(self, r_src, r_trg, mask_mat):
    # compute the nce scores for these features
    nce_scores, raw_scores, lgt_reg = \
      self._model_scores(r_src, r_trg, mask_mat)
    loss_g2l = -nce_scores.mean()
    return loss_g2l, lgt_reg

  def forward(self, r1_src_1, r5_src_1, r1_src_2, r5_src_2,
        r5_trg_1, r7_trg_1, r5_trg_2, r7_trg_2, mask_mat, mode):
    assert(mode in ['train', 'viz'])
    if mode == 'train':
      # compute costs for 1->5 prediction
      loss_1t5_1, lgt_1t5_1 = self._loss_g2l(r1_src_1, r5_trg_2[0], mask_mat)
      loss_1t5_2, lgt_1t5_2 = self._loss_g2l(r1_src_2, r5_trg_1[0], mask_mat)
      # compute costs for 1->7 prediction
      loss_1t7_1, lgt_1t7_1 = self._loss_g2l(r1_src_1, r7_trg_2[0], mask_mat)
      loss_1t7_2, lgt_1t7_2 = self._loss_g2l(r1_src_2, r7_trg_1[0], mask_mat)
      # compute costs for 5->5 prediction
      loss_5t5_1, lgt_5t5_1 = self._loss_g2l(r5_src_1, r5_trg_2[0], mask_mat)
      loss_5t5_2, lgt_5t5_2 = self._loss_g2l(r5_src_2, r5_trg_1[0], mask_mat)
      # combine costs for optimization
      loss_1t5 = 0.5 * (loss_1t5_1 + loss_1t5_2)
      loss_1t7 = 0.5 * (loss_1t7_1 + loss_1t7_2)
      loss_5t5 = 0.5 * (loss_5t5_1 + loss_5t5_2)
      lgt_reg = 0.5 * (lgt_1t5_1 + lgt_1t5_2 + lgt_1t7_1 +
                       lgt_1t7_2 + lgt_5t5_1 + lgt_5t5_2)
      return loss_1t5, loss_1t7, loss_5t5, lgt_reg
    else:
      # compute values to use for visualizations
      nce_scores, raw_scores, lgt_reg = \
        self._model_scores(r1_src_1, r7_trg_2[0], mask_mat)
      return nce_scores, raw_scores


class LossMultiNCE(nn.Module):
  '''
  Input is fixed as r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2.
  '''

  def __init__(self, tclip=10.):
    super(LossMultiNCE, self).__init__()
    self.nce_func = NCE_MI_MULTI(tclip=tclip)

  def _sample_src_ftr(self, r_cnv, subsample=False):
    # get feature dimensions
    n_batch = r_cnv.size(0)
    n_rkhs = r_cnv.size(1)
    if subsample:
      i = [torch.arange(n_batch),
           torch.randint(r_cnv.size(2), size=(n_batch,)),
           torch.randint(r_cnv.size(3), size=(n_batch,))]
      r_cnv = r_cnv[i[0],:,i[1],i[2]]
    # flatten features for use as globals in glb->lcl nce cost
    r_vec = r_cnv.reshape(n_batch, n_rkhs)
    return r_vec

  def forward(self, r1_x1, r5_x1, r7_x1, r1_x2, r5_x2, r7_x2):
    '''
    Compute nce infomax costs for various combos of source/target layers.

    Compute costs in both directions, i.e. from/to both images (x1, x2).

    rK_x1 are features from source image x1.
    rK_x2 are features from source image x2.
    '''
    # compute feature dimensions
    n_batch = int(r1_x1.size(0))
    n_rkhs = int(r1_x1.size(1))
    # make masking matrix to help compute nce costs
    mask_mat = torch.eye(n_batch)
    if torch.cuda.is_available(): mask_mat = mask_mat.cuda()

    # sample "source" features for glb->lcl predictions
    r1_src_1 = self._sample_src_ftr(r1_x1)
    r5_src_1 = self._sample_src_ftr(r5_x1, subsample=True)
    r1_src_2 = self._sample_src_ftr(r1_x2)
    r5_src_2 = self._sample_src_ftr(r5_x2, subsample=True)

    # before shape: (n_batch, n_rkhs, n_dim, n_dim)
    # after shape: (n_gpu, n_rkhs, n_batch * n_dim * n_dim)
    r5_trg_1 = r5_x1.permute(1, 0, 2, 3).reshape(1, n_rkhs, -1)
    r7_trg_1 = r7_x1.permute(1, 0, 2, 3).reshape(1, n_rkhs, -1)
    r5_trg_2 = r5_x2.permute(1, 0, 2, 3).reshape(1, n_rkhs, -1)
    r7_trg_2 = r7_x2.permute(1, 0, 2, 3).reshape(1, n_rkhs, -1)

    # compute nce for multiple infomax costs across multiple GPUs
    losses = self.nce_func(r1_src_1, r5_src_1, r1_src_2, r5_src_2,
                           r5_trg_1, r7_trg_1, r5_trg_2, r7_trg_2,
                           mask_mat, mode='train')

    losses = [l.mean() for l in losses]
    return losses


@gin.configurable
class AMDIMTask(ClassifyTask):
  def __init__(self, tclip=10, **kargs):
    super().__init__(**kargs)
    self.nce_loss = LossMultiNCE(tclip)

  def evaluate(self, sample, model_out):
    # Calculate downstream FC + MLP loss/accuracies
    results = super().evaluate(sample, model_out, multi_out=True)

    # AMDIM loss
    model_feats = model_out[1]
    loss_1t5, loss_1t7, loss_5t5, lgt_reg = self.nce_loss(*model_feats)
    amdim_loss = loss_1t5 + loss_1t7 + loss_5t5 + lgt_reg

    loss = amdim_loss + results['loss_downstream']

    results['loss_amdim'] = amdim_loss
    results['loss_1t5'] = loss_1t5
    results['loss_5t5'] = loss_5t5
    results['loss_1t7'] = loss_1t7
    results['loss_lgt_reg'] = lgt_reg

    return loss, results
