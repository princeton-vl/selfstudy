"""CMC backbone definition.

Modified from: https://github.com/HobbitLong/CMC
"""

import gin

import numpy as np
import torch
from torch import nn

from .classify import ClassifyTask, ce_loss

eps = 1e-7


# (not used anymore)
class NCECriterion(nn.Module):
  """Eq. (12): L_{NCE} """

  def __init__(self, n_data):
    super(NCECriterion, self).__init__()
    self.n_data = n_data

    # noise distribution
    self.Pn = 1 / n_data

  def forward(self, x):
    batch_size = x.shape[0]
    K = x.size(1) - 1
    Pn = self.Pn

    P_pos = x[:, 0]
    P_neg = x[:, 1:]

    # loss for positive pair
    log_D1 = torch.log(P_pos) - torch.log(P_pos + K * Pn + eps)
    # loss for K negative pair
    log_D0 = np.log(K * Pn) - torch.log(P_neg + K * Pn + eps)

    loss = -(log_D1.sum(0) + log_D0.view(-1, 1).sum(0)) / batch_size

    return loss


@gin.configurable
class CMCTask(ClassifyTask):
  def evaluate(self, sample, model_out):
    # Calculate downstream FC + MLP loss/accuracies
    results = super().evaluate(sample, model_out, multi_out=True)

    out_l, out_ab = model_out[1]
    batch_size = out_l.shape[0]
    zeros = torch.zeros([batch_size]).cuda().long()

    l_loss = ce_loss(out_l.squeeze(), zeros)
    ab_loss = ce_loss(out_ab.squeeze(), zeros)
    cmc_loss = l_loss + ab_loss

    loss = cmc_loss + results['loss_downstream']

    results['loss_l'] = l_loss
    results['loss_ab'] = ab_loss
    results['prob_l'] = out_l[:, 0].mean()
    results['prob_ab'] = out_ab[:, 0].mean()

    return loss, results
