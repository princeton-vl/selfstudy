import gin
import torch
from torch import nn
from contextlib import nullcontext

from .resnet import *
from .cmc import CMCModel
from .vae import VAEModel
from .amdim import AMDIMModel
from .rotate import RotateModel
from .amdim_ref import AMDIMRefModel

from .. import paths

backbones = {
  'resnet': ResnetModel,
  'cmc': CMCModel,
  'vae': VAEModel,
  'amdim': AMDIMModel,
  'rotate': RotateModel,
  'amdim_ref': AMDIMRefModel,
}


@gin.configurable
class Wrapper(nn.Module):
  def __init__(self,
               model_choice='resnet',
               out_classes=1000,
               run_selfsup=True,
               finetune_selfsup=True,
               run_classifier=True,
               finetune_classifier=False,
               pretrained=None,
               ignore_pretrained_fc=True,
               warmup_fc=0,
               dense_pred=False):

    super().__init__()
    self.run_selfsup = run_selfsup
    self.finetune_selfsup = finetune_selfsup
    self.run_classifier = run_classifier
    self.finetune_classifier = finetune_classifier
    self.dense_pred = dense_pred
    self.warmup_fc = warmup_fc
    self.step_count = 0

    # Set up model
    self.backbone = backbones[model_choice](run_selfsup=run_selfsup)

    # Output classification layer
    if run_classifier:
      if not isinstance(out_classes, list):
        out_classes = [out_classes]

      in_f = self.backbone.out_feats
      self.out_fc = nn.ModuleList([nn.Linear(in_f, f)
                                   for f in out_classes])
      self.out_mlp = nn.ModuleList([nn.Sequential(nn.Linear(in_f, in_f, bias=False),
                                                  nn.BatchNorm1d(in_f), nn.ReLU(True),
                                                  nn.Linear(in_f, f))
                                    for f in out_classes])

    if pretrained is not None:
      # Load pretrained weights
      pretrained_weights = torch.load(f'{paths.EXP_DIR}/{pretrained}/snapshot')
      print("Loading pretrained model:", f'{paths.EXP_DIR}/{pretrained}/snapshot')

      if ignore_pretrained_fc:
        print("Ignoring FC weights from pretrained snapshot.")
        to_remove = []
        for k in pretrained_weights['model']:
          if 'out_fc' in k or 'out_mlp' in k:
            to_remove += [k]
        for k in to_remove:
          del pretrained_weights['model'][k]

      self.load_state_dict(pretrained_weights['model'], strict=False)

  def forward(self, *args, **kargs):
    class_pred = None
    self.step_count += 1

    # No need to track gradients if not finetuning backbone
    update_backbone = self.finetune_classifier or self.finetune_selfsup
    cm = nullcontext() if update_backbone else torch.no_grad()

    # Get model output
    with cm:
      feats, dense_feats, s_out = self.backbone(*args, **kargs)

    if self.dense_pred:
      # Flatten dense per-pixel features for FC/MLP processing
      feats = dense_feats.permute(0,2,3,1).reshape(-1, dense_feats.shape[1])

    if not self.finetune_selfsup and s_out is not None:
      if isinstance(s_out, list) or isinstance(s_out, tuple):
        s_out = [s.detach() for s in s_out]
      else:
        s_out = s_out.detach()

    if self.run_classifier:
      warming_up = self.step_count < self.warmup_fc
      if warming_up or not self.finetune_classifier:
        feats = feats.detach()

      fc_pred = [fc(feats) for fc in self.out_fc]
      mlp_pred = [mlp(feats) for mlp in self.out_mlp]
      class_pred = [fc_pred, mlp_pred]

    if self.dense_pred:
      # Reshape flattened output to dense per-pixel output
      feats = dense_feats
      if class_pred is not None:
        d = feats.shape
        class_pred = [[p.reshape(d[0],d[2],d[3],-1).permute(0,3,1,2)
                       for p in tmp_pred] for tmp_pred in class_pred]

    return class_pred, s_out, feats
