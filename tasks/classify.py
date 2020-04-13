import gin

import torch
from torch.functional import F

from .depth import depth_loss_and_accuracy

ce_loss = torch.nn.CrossEntropyLoss()


def class_loss_and_accuracy(pred, label, is_dense=False):
  if is_dense:
    out_res = pred.shape[2]
    label = F.interpolate(label.float(), size=out_res)[:,1].long()

  loss = ce_loss(pred, label)
  acc = (pred.argmax(1) == label).float().mean()

  return loss, acc


def multi_loss_and_accuracy(pred, label, is_dense=False):
  if pred is None:
    # Return empty results
    return {k: torch.zeros(1) for k in [
      'loss_fc_cls', 'accuracy', 'loss_mlp_cls', 'acc_mlp_cls',
      'loss_fc_3d', 'acc_fc_3d', 'loss_mlp_3d', 'acc_fc_3d',
      'loss_downstream'
    ]}

  fc_pred = pred[0]
  mlp_pred = pred[1]
  r = {}

  predict_3d = label.ndim > 1

  if is_dense:
    out_res = pred[0][0].shape[2]
    label = F.interpolate(label.float(), size=out_res)
    class_label = label[:,1].long()
    depth_label = label[:,2]
    depth_label[depth_label == 0] = depth_label.max()

  elif predict_3d:
    class_label = label[:,0]
    pose_label = label[:,1]

  else:
    class_label = label

  class_fc, pose_fc = fc_pred
  class_mlp, pose_mlp = mlp_pred

  r['loss_fc_cls'], r['accuracy'] = class_loss_and_accuracy(class_fc, class_label)
  r['loss_mlp_cls'], r['acc_mlp_cls'] = class_loss_and_accuracy(class_mlp, class_label)

  if is_dense:
    r['loss_fc_3d'], r['acc_fc_3d'], _, _ = depth_loss_and_accuracy(fc_pred[1][:,0].float(), depth_label / 25)
    r['loss_mlp_3d'], r['acc_mlp_3d'], _, _ = depth_loss_and_accuracy(mlp_pred[1][:,0].float(), depth_label / 25)
  elif predict_3d:
    r['loss_fc_3d'], r['acc_fc_3d'] = class_loss_and_accuracy(pose_fc, pose_label)
    r['loss_mlp_3d'], r['acc_mlp_3d'] = class_loss_and_accuracy(pose_mlp, pose_label)

  loss_keys = [k for k in r if 'loss' in k]
  r['loss_downstream'] = sum(r[k] for k in loss_keys) / len(loss_keys)

  return r


@gin.configurable
class ClassifyTask:
  def __init__(self, is_dense=False, is_pose=False):
    try:
      is_dense = gin.query_parameter('Wrapper.dense_pred')
    except:
      pass

    self.is_dense = is_dense
    self.is_pose = is_pose

  def forward(self, model, sample):
    inp_img, _, idxs = sample
    return model(inp_img, idxs)

  def evaluate(self, sample, model_out, multi_out=False):
    class_pred = model_out[0]
    label = sample[1]

    if not multi_out:
      out_idx = 1 if self.is_pose else 0
      class_pred = class_pred[0][out_idx]

      if label.ndim > 1 and not self.is_dense:
        label = label[:, out_idx]

      loss, class_acc = class_loss_and_accuracy(class_pred, label, self.is_dense)
      return loss, {'accuracy': class_acc}

    else:
      return multi_loss_and_accuracy(class_pred, label, self.is_dense)
