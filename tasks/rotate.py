import gin

import torch
from torch.functional import F

from .classify import ClassifyTask, class_loss_and_accuracy
from ..util import augment as aug


rotate_tf = aug.Rotate90()


@gin.configurable
class RotateTask(ClassifyTask):
  def forward(self, model, sample):
    imgs, targets, idxs = sample
    imgs_, targets_, tfs = aug.apply_tf(imgs, rotate_tf, targets, self.is_dense)
    self.applied_rot = torch.LongTensor([t['choice'] for t in tfs])
    self.rotated_targets = targets_

    return model(imgs_, idxs)

  def evaluate(self, sample, model_out):
    if self.rotated_targets is not None:
      # Update ground truth if we augmented dense target maps
      sample[1] = self.rotated_targets

    # Calculate downstream FC + MLP loss/accuracies
    results = super().evaluate(sample, model_out, multi_out=True)

    # Calculate rotation loss
    rotate_pred = model_out[1]
    rotate_label = self.applied_rot.to(rotate_pred.device)
    rotate_loss, rotate_acc = class_loss_and_accuracy(rotate_pred,
                                                      rotate_label)

    loss = rotate_loss + results['loss_downstream']
    results['loss_rotate'] = rotate_loss
    results['acc_rotate'] = rotate_acc

    return loss, results
