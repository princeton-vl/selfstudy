import gin
import math
import numpy as np
import torch
from torch import nn
from torch.functional import F

from .resnet import init_resnet


class NCEAverage(nn.Module):

  def __init__(self, input_size, output_size, K, T=0.07, momentum=0.5):
    super().__init__()
    self.K = K

    self.register_buffer('params', torch.tensor([K, T, -1, -1, momentum]))
    stdv = 1. / math.sqrt(input_size / 3)
    self.register_buffer('memory_l', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))
    self.register_buffer('memory_ab', torch.rand(output_size, input_size).mul_(2 * stdv).add_(-stdv))

  def forward(self, l, ab, y, idx=None):
    K = int(self.params[0].item())
    T = self.params[1].item()
    Z_l = self.params[2].item()
    Z_ab = self.params[3].item()
    momentum = self.params[4].item()

    batch_size = l.size(0)
    output_size = self.memory_l.size(0)
    input_size = self.memory_l.size(1)

    # sample random negative idxs
    if idx is None:
      idx = torch.randint(output_size, [batch_size * (K + 1)]).view(batch_size, -1).cuda()
      idx[:,0].copy_(y.data)
    # sample
    weight_l = torch.index_select(self.memory_l, 0, idx.view(-1)).detach()
    weight_l = weight_l.view(batch_size, K + 1, input_size)
    out_ab = torch.bmm(weight_l, ab.view(batch_size, input_size, 1))
    # sample
    weight_ab = torch.index_select(self.memory_ab, 0, idx.view(-1)).detach()
    weight_ab = weight_ab.view(batch_size, K + 1, input_size)
    out_l = torch.bmm(weight_ab, l.view(batch_size, input_size, 1))

    out_ab = torch.div(out_ab, T).contiguous()
    out_l = torch.div(out_l, T).contiguous()

    # # update memory
    with torch.no_grad():
      l_pos = torch.index_select(self.memory_l, 0, y.view(-1))
      l_pos.mul_(momentum)
      l_pos.add_(torch.mul(l, 1 - momentum))
      l_norm = l_pos.pow(2).sum(1, keepdim=True).pow(0.5)
      updated_l = l_pos.div(l_norm)
      self.memory_l.index_copy_(0, y, updated_l)

      ab_pos = torch.index_select(self.memory_ab, 0, y.view(-1))
      ab_pos.mul_(momentum)
      ab_pos.add_(torch.mul(ab, 1 - momentum))
      ab_norm = ab_pos.pow(2).sum(1, keepdim=True).pow(0.5)
      updated_ab = ab_pos.div(ab_norm)
      self.memory_ab.index_copy_(0, y, updated_ab)

    return out_l, out_ab


@gin.configurable
class CMCModel(nn.Module):
  def __init__(self,
               dataset_size=1281167,
               run_selfsup=True,
               nce_k=4096,
               nce_t=0.07,
               nce_m=0.5):
    super().__init__()

    try:
      tmp_size = gin.query_parameter('BaseDataset.semisupervised')
      if tmp_size is not None:
        dataset_size = tmp_size
        print("Set CMCModel.dataset_size to", dataset_size)
    except:
      pass

    self.run_selfsup = run_selfsup
    self.resnet_l = init_resnet(in_channels=1)
    self.resnet_ab = init_resnet(in_channels=2)

    feat_dim = self.resnet_l.out_feats
    self.out_feats = feat_dim * 2

    if run_selfsup:
      self.contrast = NCEAverage(feat_dim, dataset_size, nce_k, nce_t, nce_m)

  def forward(self, x, idx):
    contrast_out = None
    l, ab = torch.split(x, [1, 2], dim=1)

    feat_l, all_feats_l = self.resnet_l(l)
    feat_ab, all_feats_ab = self.resnet_ab(ab)

    # Normalize features
    feat_l = F.normalize(feat_l)
    feat_ab = F.normalize(feat_ab)

    feats = torch.cat([feat_l, feat_ab], 1)
    dense_feats = torch.cat([all_feats_l[-2],
                             all_feats_ab[-2]], 1)

    if self.run_selfsup:
      self.contrast.float()
      contrast_out = self.contrast(feat_l.float(), feat_ab.float(), idx)

    return feats, dense_feats, contrast_out
