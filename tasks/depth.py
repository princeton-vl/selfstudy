import gin

import torch
from torch.functional import F

l1_loss = torch.nn.L1Loss()
l2_loss = torch.nn.MSELoss()


def depth_loss_and_accuracy(pred, label, loss_fn=l1_loss):
  loss = loss_fn(pred, label)
  d, _ = torch.stack([pred / label, label / pred]).max(0)
  d_thr = (d < 1.25).float().mean()
  d_strict = (d < 1.1).float().mean()
  rmse = l2_loss(pred, label).sqrt()

  return loss, d_thr, d_strict, rmse


@gin.configurable
class DepthTask:
  def __init__(self, is_dense=True, use_l2=False):
    try:
      is_dense = gin.query_parameter('Wrapper.dense_pred')
    except:
      pass

    assert is_dense, "Depth estimation should only be run with dense predictions."

    self.is_dense = is_dense
    self.use_l2 = use_l2

  def forward(self, model, sample):
    inp_img, _, idxs = sample
    return model(inp_img, idxs)

  def evaluate(self, sample, model_out):
    depth_pred = model_out[1][0][1]
    label = sample[1]
    loss_fn = l2_loss if self.use_l2 else l1_loss

    # Resize ground-truth appropriately
    out_res = depth_pred.shape[2]
    label = F.interpolate(label.float(), size=out_res)[:,2]

    # Remap 0 to max dist
    label[label == 0] = label.max()

    loss, d_thr, d_strict, rmse = depth_loss_and_accuracy(depth_pred[:,0].float(),
                                                          label / 25, loss_fn=loss_fn)

    return loss, {'accuracy': d_thr, 'depth_loss': loss,
                  'rmse': rmse, 'd_strict': d_strict}
