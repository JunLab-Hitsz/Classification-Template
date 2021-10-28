import numpy as np
import torch
import os,shutil

def augmentation(x, max_shift=2):
  _, _, height, width = x.size()

  h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
  source_height_slice = slice(max(0, h_shift), h_shift + height)
  source_width_slice = slice(max(0, w_shift), w_shift + width)
  target_height_slice = slice(max(0, -h_shift), -h_shift + height)
  target_width_slice = slice(max(0, -w_shift), -w_shift + width)

  shifted_image = torch.zeros(*x.size())
  shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
  return shifted_image.float()

def print_log(print_string, logger):
  print('{}'.format(print_string))
  logger.log('{}\n'.format(print_string))

def save_checkpoint(state, is_best, save_path, filename):
  filename = os.path.join(save_path, filename)
  torch.save(state, filename)
  if is_best:
    bestname = os.path.join(save_path, 'model_best.pth.tar')
    shutil.copyfile(filename, bestname)

class RecorderMeter(object):
  """Computes and stores the minimum loss value and its epoch index"""
  def __init__(self, total_epoch):
    self.reset(total_epoch)

  def reset(self, total_epoch):
    assert total_epoch > 0
    self.total_epoch   = total_epoch
    self.current_epoch = 0
    self.epoch_losses  = np.zeros((self.total_epoch+1, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_losses  = self.epoch_losses - 1

    self.epoch_accuracy= np.zeros((self.total_epoch+1, 2), dtype=np.float32) # [epoch, train/val]
    self.epoch_accuracy= self.epoch_accuracy
    self.best_accuracy = 0
    self.best_accuracy_epoch = 0

  def update(self, idx, train_loss, train_acc, val_loss, val_acc):
    assert idx > 0 and idx <= self.total_epoch, 'total_epoch : {} , but update with the {} index'.format(self.total_epoch, idx)
    self.epoch_losses  [idx, 0] = train_loss
    self.epoch_losses  [idx, 1] = val_loss
    self.epoch_accuracy[idx, 0] = train_acc
    self.epoch_accuracy[idx, 1] = val_acc
    self.current_epoch = idx + 1
    if val_acc > self.best_accuracy:
      self.best_accuracy = val_acc
      self.best_accuracy_epoch = idx
      return True
    else:
      return False

def get_learning_rate(optimizer):
  '''
  Returns the current LR from the optimizer module.
  '''

  for params in optimizer.param_groups:
    return params['lr']

def print_log(print_string, log):
  print("{}".format(print_string))
  log.write('{}\n'.format(print_string))
  log.flush()