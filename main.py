from __future__ import division

from numpy import record
from torch.functional import norm

import sys,argparse,time,os

sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, StepLR
from utils import save_checkpoint, RecorderMeter, get_learning_rate
from utils import print_log as print_helper
from lr_scheduler import WarmUpLR
from loss_criterion import CapsuleLossCrossEntropy, CapsuleLossMarginLoss
import models
from torch.utils.tensorboard import SummaryWriter

# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='Trains ConvCapsule Networks', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--warmup_epoch', type=int, default=1, help='Number of epoch to warmup.')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=1e-1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--milestones', type=int, nargs='+', default=[30,60,80], help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.2, help='LR is multiplied by gamma on schedule')
parser.add_argument('--epsilon', type=float, default=1e-5, help='avoid zero divided during capsules normalize')
parser.add_argument('--data_path', type=str, default='/opt/datasets', help='The path to the datasets')
parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10', 'cifar100', 'svhn','fmnist', 'imagenet'], help='Choose between Cifar10/100 and ImageNet.')
# Checkpoints
parser.add_argument('--resume', default='epochs/2021-10-26/14:43:02/checkpoint.pth.tar', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--save_path', type=str, default='./epochs', help='Folder to save checkpoints and log.')
# Acceleration
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
parser.add_argument('--exp_name', type=str, default='', help='The name of the experiment.')
parser.add_argument('--reconstruction', type=int, default=1, help='The reconstruction module')
# random seed
args = parser.parse_args()
reconstruction = False
if args.reconstruction == 1:
    reconstruction = True

time_tuple = time.localtime(time.time())
if args.exp_name != '':
    exp_name = args.exp_name
    exp_name = args.dataset + '_' + exp_name
else:
    exp_name = time.strftime("%Y-%m-%d %H:%M:%S", time_tuple)
args.save_path = os.path.join(args.save_path, *(exp_name.split(' ')))


log_path = os.path.join('./logs', *(exp_name.split(' ')))
if not os.path.isdir(log_path):
    os.makedirs(log_path)
writer = SummaryWriter(log_path)
log = open(os.path.join(log_path, 'log.log'), 'w')

def print_log(print_string):
    print_helper(print_string=print_string, log=log)

def tb_log(text_string, tag='log', global_step=None):
    writer.add_text(tag=tag, text_string=text_string, global_step=global_step)
    print_log(text_string)

if __name__ == "__main__":
    from torch.autograd import Variable
    from torch.optim import SGD, Adam
    from torchnet.engine import Engine
    from torchvision.utils import make_grid
    from tqdm import tqdm
    import torchnet as tnt
    import load_data

    if args.dataset == 'cifar10':
       train_loader, test_loader, img_size, num_class = load_data.cifar10(args.data_path, args.batch_size, True)
    elif args.dataset == 'svhn':
       train_loader, test_loader, img_size, num_class = load_data.svhn(args.data_path, args.batch_size, True)
    elif args.dataset == 'fmnist':
       train_loader, test_loader, img_size, num_class = load_data.fmnist(args.data_path, args.batch_size, True)
    elif args.dataset == 'mnist':
       train_loader, test_loader, img_size, num_class = load_data.mnist(args.data_path, args.batch_size, True)
    else: raise ValueError('Nonsupported datases, {:s}'.format(args.dataset))

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
    confusion_meter = tnt.meter.ConfusionMeter(num_class, normalized=True)

    recorder = RecorderMeter(args.epochs)

    criterion = nn.CrossEntropyLoss()

    
    # model = models.DeepCapsModel(num_class=num_class,in_channels=1, img_height=img_size, img_width=img_size, activation=CAPSULE_RELU)
    model = models.CNN()

    model.cuda()
    if args.resume:
        if os.path.isfile(args.resume):
            tb_log(text_string="=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
            args.learning_rate = 1e-2
            tb_log(text_string="=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']))
        else:
            tb_log(text_string="=> no checkpoint found at '{}'".format(args.resume))
    else:
        tb_log(text_string="=> do not use any checkpoint")

    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay, nesterov=True)
    scheduler = MultiStepLR(optimizer, args.milestones ,gamma=args.gamma)

    args_state = {k:v for k,v in args._get_kwargs()}
    tb_log(text_string=str(args_state))
    tb_log(text_string="# parameters:{}".format(sum(param.numel() for param in model.parameters())))
    tb_log(text_string="model:\n{}".format(model))

    def get_iterator(mode):
        if mode: return train_loader
        else: return test_loader

    def processor(sample):
        data, labels, training = sample
        data = Variable(data).cuda()
        labels = Variable(labels).cuda()

        logits = model(data)
        loss = capsule_loss(labels, logits)
        assert torch.isnan(loss).sum() == 0, tb_log(text_string='The loss is nan!') 

        return loss, logits


    def reset_meters():
        meter_accuracy.reset()
        meter_loss.reset()
        confusion_meter.reset()


    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        meter_accuracy.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        confusion_meter.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].item())



    def on_start_epoch(state):
        reset_meters()
        state['iterator'] = tqdm(state['iterator'])


    def on_end_epoch(state):
        train_acc = meter_accuracy.value()[0]
        train_los = meter_loss.value()[0]
        current_lr = get_learning_rate(optimizer)
        tb_log(text_string='[Epoch %d/%d] Learning Rate: %.8f' % (state['epoch'], args.epochs, current_lr), global_step=state['epoch'])

        tb_log(text_string='[Epoch %d/%d] Training Loss: %.4f (Accuracy: %.2f%%)' % (
            state['epoch'], args.epochs, train_los, train_acc), global_step=state['epoch'])

        writer.add_scalar('Train Loss', train_los, state['epoch'])
        writer.add_scalar('Train Accuracy', train_acc, state['epoch'])

        # for name, param in model.named_parameters():
        #     writer.add_histogram(name, param.clone().cpu().data.numpy(), state['epoch'])
        reset_meters()

        engine.test(processor, get_iterator(False))
        val_acc = meter_accuracy.value()[0]
        val_los = meter_loss.value()[0]
        writer.add_scalar('Test Loss', val_los, state['epoch'])
        writer.add_scalar('Test Accuracy', val_acc, state['epoch'])

        is_best = recorder.update(state['epoch'], train_los, train_acc, val_los, val_acc)
        tb_log(text_string='[Epoch %d/%d] Testing Loss: %.4f (Accuracy: %.2f%%) \n The best test Accuracy: %.2f%% at %d Epoch'% (state['epoch'], args.epochs, val_los, val_acc, recorder.best_accuracy, recorder.best_accuracy_epoch), global_step=state['epoch'])
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)

        save_checkpoint({
            'epoch': state['epoch'],
            'optimizer': state['optimizer'],
            'state_dict': model.state_dict(),
            'recorder': recorder
        }, is_best, args.save_path, 'checkpoint.pth.tar')


        scheduler.step()

    def on_start(state):
        state['epoch'] = args.start_epoch

    def on_update(state):
        pass
    
    engine.hooks['on_start'] = on_start
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, get_iterator(True), maxepoch=args.epochs, optimizer=optimizer)
