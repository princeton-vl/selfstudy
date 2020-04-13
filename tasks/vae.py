import gin

import torch
from torch import nn
from torch.functional import F

from .classify import ClassifyTask


@gin.configurable
class VAETask(ClassifyTask):
  def __init__(self, lmd=.001, **kargs):
    super().__init__(**kargs)
    self.lmd = lmd

  def evaluate(self, sample, model_out):
    # Calculate downstream FC + MLP loss/accuracies
    results = super().evaluate(sample, model_out, multi_out=True)

    inp_img = sample[0]
    out_img, mu, logvar = [tmp_out.float() for tmp_out in model_out[1]]
    out_res = out_img.shape[2]

    # Reconstruction loss
    target_img = F.interpolate(inp_img.float(), size=out_res)
    MSE = F.mse_loss(target_img, out_img)

    # KL divergence
    if self.lmd > 0:
      KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    else:
      KLD = torch.Tensor([0])[0]

    vae_loss = MSE + self.lmd * KLD

    loss = vae_loss + results['loss_downstream']
    results['loss_vae'] = vae_loss
    results['loss_mse'] = MSE
    results['loss_kld'] = KLD

    return loss, results
