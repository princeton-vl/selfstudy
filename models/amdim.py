import gin
import math
import numpy as np
import torch
from torch import nn
from torch.functional import F

from .resnet import ResnetModel

@gin.configurable
class AMDIMModel(ResnetModel):
  def __init__(self, run_selfsup=True):
    super().__init__()
    self.run_selfsup = run_selfsup

    self.layer_idx_ref = [-7,-4]
    if self.resnet.n_layers == 3:
      self.layer_idx_ref = [-5,-4]

    in_f = self.resnet.fc.in_features
    out_f = self.out_feats
    self.rkhs_7 = nn.Conv2d(in_f // 2, out_f, 1)
    self.rkhs_5 = nn.Conv2d(in_f, out_f, 1)
    self.pool_7 = nn.AdaptiveAvgPool2d(7)
    self.pool_5 = nn.AdaptiveAvgPool2d(5)

  def forward(self, x, idx):
    selfsup_out = None
    merge_dims = x.dim() == 5
    if merge_dims:
      d0, d1 = x.shape[:2]

    feats, dense_feats, all_feats = super().forward(x, idx)

    if self.run_selfsup:
      r7 = self.pool_7(self.rkhs_7(all_feats[self.layer_idx_ref[0]]))
      r7 = r7.view(d0, d1, *r7.shape[1:])
      r5 = self.pool_5(self.rkhs_5(all_feats[self.layer_idx_ref[1]]))
      r5 = r5.view(d0, d1, *r5.shape[1:])
      selfsup_out = (feats[:,0].unsqueeze(2).unsqueeze(3), r5[:,0], r7[:,0],
                     feats[:,1].unsqueeze(2).unsqueeze(3), r5[:,1], r7[:,1])

    if feats.dim() == 3:
      feats = feats[:,0]
      dense_feats = dense_feats[:,0]

    return feats, dense_feats, selfsup_out
