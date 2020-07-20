import argparse
import os
import shutil
import time
from datetime import datetime
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from peleenet import PeleeNet
from profiling import Profiling
# from torch import itt
import ilit

model_names = [ 'peleenet']

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='peleenet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: peleenet)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--epochs', default=120, type=int, metavar='N',
                    help='number of total epochs to run (default: 120')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 512)')
parser.add_argument('--lr', '--learning-rate', default=0.18, type=float,
                    metavar='LR', help='initial learning rate (default: 0.18)')
parser.add_argument('--lr-policy', metavar='POLICY', default='cosine',
                    choices=['cosine','step'],
                    help='learning rate policy: cosine | step (default: cosine)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--input-dim', default=224, type=int,
                    help='size of the input dimension (default: 224)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--int8', action='store_true', help='int8 quantization')
parser.add_argument('--profile', default='none', type=str, help='Profile')

best_acc1 = 0


def main():
    global args, best_acc1
    args = parser.parse_args()
    print( 'args:',args)

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # Val data loading
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(args.input_dim+32),
            transforms.CenterCrop(args.input_dim),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.RandomSampler(val_dataset, replacement=True, num_samples=1280))

    num_classes = len(val_dataset.classes)
    print('Total classes: ',num_classes)

    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'peleenet':
        model = PeleeNet(num_classes=num_classes)
    else:
        print("=> unsupported model '{}'. creating PeleeNet by default.".format(args.arch))
        model = PeleeNet(num_classes=num_classes)

    if args.distributed:
        # model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)
    # else:
    #     # DataParallel will divide and allocate batch_size to all available GPUs
    #     # model = torch.nn.DataParallel(model).cuda()
    #     model = torch.nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    elif args.pretrained:
        if os.path.isfile('checkpoint.pth.tar'):
            checkpoint = torch.load('checkpoint.pth.tar', map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}' (epoch {}, acc@1 {})"
                  .format(args.pretrained, checkpoint['epoch'], checkpoint['best_acc1']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
        mode = model.to(torch.device('cpu'))

    # cudnn.benchmark = True


    with Profiling(model, os.getpid(), enabled=False):
        if args.evaluate:
            model.eval()
            if args.int8:
                # print('Before fuse')
                # print(model)
                model.fuse()
                # print('After fuse')
                # print(model)
                # model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                # # print(model.qconfig)
                # torch.quantization.prepare(model, inplace=True)
                # print('validate for converting')
                # # itt.range_push('validate_convert')
                # validate(val_loader, model, criterion)
                # # itt.range_pop()
                # # itt.range_push('quant_convert')
                # torch.quantization.convert(model, inplace=True)
                # # itt.range_pop()
                tuner = ilit.Tuner('./config.yaml')
                model = tuner.tune(model, val_loader, eval_dataloader=val_loader)
            print('main validation')
            # itt.range_push('main validation')
            validate(val_loader, model, criterion, profile=args.profile)
            # itt.range_pop()
        return

    # Training data loading
    traindir = os.path.join(args.data, 'train')

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(args.input_dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion)

        # remember best Acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        ### Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch,  args.epochs, args.lr, iteration=i,
                                  iterations_per_epoch=len(train_loader),
                                  method=args.lr_policy)


        # measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))



def validate(val_loader, model, criterion, profile='none'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # target = target.cuda(async=True)
            # input_var = torch.autograd.Variable(input)
            # target_var = torch.autograd.Variable(target)

            # compute output
            end = time.time()
            if profile != 'none':
                with torch.autograd.profiler.profile() as prof:
                    output = model(input)
                if profile == 'stdio':
                    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
                else:
                    if not os.path.exists('LOGS'):
                        os.mkdir('LOGS')
                    prof.export_chrome_trace('LOGS/'+profile+'.json')
            else:
                num = 0
                dur_sum = 0
                for i in range(0, 20):
                    t0 = time.time()
                    output = model(input)
                    if i >= 10:
                        num += 1
                        dur_sum += time.time() - t0

            # measure elapsed time
            batch_time.update(time.time() - end)
            # end = time.time()

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # if i % args.print_freq == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            #           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            #            i, len(val_loader), batch_time=batch_time, loss=losses,
            #            top1=top1, top5=top5))

        # print('Time {batch_time.avg:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
        #         .format(batch_time=batch_time, top1=top1, top5=top5))
        print('time: {:.3f}s, Acc1: {:.3f}, Acc5: {:.3f}'.format(dur_sum/num, top1.avg, top5.avg))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, num_epochs, init_lr, iteration=None,
                         iterations_per_epoch=None, method='step'):
    if method == 'cosine':
        T_total = num_epochs * iterations_per_epoch
        T_cur = (epoch % num_epochs) * iterations_per_epoch + iteration
        lr = 0.5 * init_lr * (1 + math.cos(math.pi * T_cur / T_total))
    else:
        lr = init_lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    # with torch.autograd.profiler.emit_itt(enabled=True):
    main()
