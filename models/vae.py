import gin
import numpy as np

import torch
from torch import nn

import torchvision.models.resnet as rn
from torchvision import transforms

from .resnet import ResnetModel

@gin.configurable
class VAEModel(ResnetModel):
  def __init__(self, run_selfsup=True, output_res=64,
               inpainting=False, do_reparameterize=True):
    super().__init__()
    self.run_selfsup = run_selfsup
    self.inpainting = inpainting
    self.do_reparameterize = do_reparameterize

    f = self.out_feats

    # Define decoding layers
    norm_layer = nn.BatchNorm2d

    self.fc1 = nn.Linear(f, f)
    self.fc2 = nn.Linear(f, f)

    n_scales = int(np.log2(output_res)) - 1
    layers = []
    f_amts = [f] + [2 ** min(8, 10-i) for i in range(n_scales)]
    for i in range(n_scales):
      layers += [
        nn.ConvTranspose2d(f_amts[i], f_amts[i+1], 4, 2, min(i, 1)),
        nn.BatchNorm2d(f_amts[i+1]), nn.ReLU(),
      ]
    layers += [nn.Conv2d(f_amts[-1], 3, 1)]

    self.decoder = nn.Sequential(*layers)
    self.erase_tf = transforms.RandomErasing(value='random')

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return mu + eps*std

  def forward(self, x, idx):
    out_img, mu, logvar = None, None, None

    if self.inpainting:
      x = torch.stack([self.erase_tf(x_) for x_ in x])

    feats, all_feats = self.resnet(x)
    dense_feats = all_feats[-2]

    if self.run_selfsup:
      mu, logvar = self.fc1(feats), self.fc2(feats)

      if self.do_reparameterize:
        z = self.reparameterize(mu, logvar)
      else:
        z = mu

      out_img = self.decoder(z.view(-1, self.out_feats, 1, 1))

    return feats, dense_feats, (out_img, mu, logvar)
