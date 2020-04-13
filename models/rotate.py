import gin
import math
import numpy as np
import torch
from torch import nn
from torch.functional import F

from .resnet import ResnetModel

@gin.configurable
class RotateModel(ResnetModel):
  def __init__(self, run_selfsup=True):
    super().__init__()
    self.run_selfsup = run_selfsup
    self.rotate_fc = nn.Linear(self.out_feats, 4)

  def forward(self, x, idx):
    feats, dense_feats, _ = super().forward(x, idx)

    rotate_pred = None
    if self.run_selfsup:
      rotate_pred = self.rotate_fc(feats)

    return feats, dense_feats, rotate_pred
