import warnings
warnings.simplefilter("ignore", (UserWarning, FutureWarning))

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
from tqdm import tqdm

from utils import logger
from utils import data_utils
from utils import augmentation as aug
from utils import metrics
from models import unet

import torch
import torch.optim as optim
import time
import argparse
import shutil
import os


def main(data_path, batch_size, num_epochs, start_epoch, learning_rate, momentum, print_freq, run, resume, data_set):
    """

    Args:
        data_path:
        batch_size:
        num_epochs:

    Returns:

    """
    since = time.time()

    # get model
    model = unet.UNetSmall()

    if torch.cuda.is_available():
        model = model.cuda()

    # set up binary cross entropy and dice loss
    criterion = metrics.BCEDiceLoss()

    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # decay LR
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=18, gamma=0.1)

    # starting params
    best_loss = 999

    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)

            if checkpoint['epoch'] > start_epoch:
                start_epoch = checkpoint['epoch']

            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # get data
    mass_dataset_train = data_utils.MassRoadBuildingDataset(data_path, data_set, 'train',
                                                       transform=transforms.Compose([aug.ToTensorTarget()]))

    mass_dataset_val = data_utils.MassRoadBuildingDataset(data_path, data_set, 'valid',
                                                     transform=transforms.Compose([aug.ToTensorTarget()]))

    # creating loaders
    train_dataloader = DataLoader(mass_dataset_train, batch_size=batch_size, num_workers=2, shuffle=True)
    val_dataloader = DataLoader(mass_dataset_val, batch_size=3, num_workers=2, shuffle=False)

    # loggers
    train_logger = logger.Logger('../logs/run_{}/training'.format(str(run)), print_freq)
    val_logger = logger.Logger('../logs/run_{}/validation'.format(str(run)), print_freq)

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # step the learning rate scheduler
        lr_scheduler.step()

        # run training and validation
        train_metrics = train(train_dataloader, model, criterion, optimizer, lr_scheduler, train_logger, epoch)
        valid_metrics = validation(val_dataloader, model, criterion, val_logger, epoch)

        # store best loss and save a model checkpoint
        is_best = valid_metrics['valid_loss'] < best_loss
        best_loss = min(valid_metrics['valid_loss'], best_loss)
        save_checkpoint({
            'epoch': epoch,
            'arch': 'UNetSmall',
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict()
        }, is_best)

        cur_elapsed = time.time() - since
        print('Current elapsed time {:.0f}m {:.0f}s'.format(cur_elapsed // 60, cur_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def train(train_loader, model, criterion, optimizer, scheduler, logger, epoch_num):
    """

    Args:
        train_loader:
        model:
        criterion:
        optimizer:
        epoch:

    Returns:

    """
    # logging accuracy and loss
    train_acc = metrics.MetricTracker()
    train_loss = metrics.MetricTracker()

    log_iter = len(train_loader)//logger.print_freq

    scheduler.step()

    # iterate over data
    for idx, data in enumerate(tqdm(train_loader, desc="training")):

        # get the inputs and wrap in Variable
        if torch.cuda.is_available():
            inputs = Variable(data['sat_img'].cuda())
            labels = Variable(data['map_img'].cuda())
        else:
            inputs = Variable(data['sat_img'])
            labels = Variable(data['map_img'])

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()
        outputs = model(inputs)
        outputs = torch.nn.functional.sigmoid(outputs)

        loss = criterion(outputs, labels)

        # backward
        loss.backward()
        optimizer.step()

        train_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        train_loss.update(loss.data[0], outputs.size(0))

        # tensorboard logging
        if idx % log_iter == 0:

            step = (epoch_num*logger.print_freq)+(idx/log_iter)

            # log accuracy and loss
            info = {
                'loss': train_loss.avg,
                'accuracy': train_acc.avg
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)

            # log weights, biases, and gradients
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), step)
                logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step)

            # log the sample images
            log_img = [data_utils.show_tensorboard_image(data['sat_img'], data['map_img'], outputs, as_numpy=True),]
            logger.image_summary('train_images', log_img, step)

    print('Training Loss: {:.4f} Acc: {:.4f}'.format(train_loss.avg, train_acc.avg))
    print()

    return {'train_loss': train_loss.avg, 'train_acc': train_acc.avg}


def validation(valid_loader, model, criterion, logger, epoch_num):
    """

    Args:
        train_loader:
        model:
        criterion:
        optimizer:
        epoch:

    Returns:

    """
    # logging accuracy and loss
    valid_acc = metrics.MetricTracker()
    valid_loss = metrics.MetricTracker()

    log_iter = len(valid_loader)//logger.print_freq

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for idx, data in enumerate(tqdm(valid_loader, desc='validation')):

        # get the inputs and wrap in Variable
        if torch.cuda.is_available():
            inputs = Variable(data['sat_img'].cuda(), volatile=True)
            labels = Variable(data['map_img'].cuda(), volatile=True)
        else:
            inputs = Variable(data['sat_img'], volatile=True)
            labels = Variable(data['map_img'], volatile=True)

        # forward
        # prob_map = model(inputs) # last activation was a sigmoid
        # outputs = (prob_map > 0.3).float()
        outputs = model(inputs)
        outputs = torch.nn.functional.sigmoid(outputs)

        loss = criterion(outputs, labels)

        valid_acc.update(metrics.dice_coeff(outputs, labels), outputs.size(0))
        valid_loss.update(loss.data[0], outputs.size(0))

        # tensorboard logging
        if idx % log_iter == 0:

            step = (epoch_num*logger.print_freq)+(idx/log_iter)

            # log accuracy and loss
            info = {
                'loss': valid_loss.avg,
                'accuracy': valid_acc.avg
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)

            # log the sample images
            log_img = [data_utils.show_tensorboard_image(data['sat_img'], data['map_img'], outputs, as_numpy=True),]
            logger.image_summary('valid_images', log_img, step)

    print('Validation Loss: {:.4f} Acc: {:.4f}'.format(valid_loss.avg, valid_acc.avg))
    print()

    return {'valid_loss': valid_loss.avg, 'valid_acc': valid_acc.avg}


# create a function to save the model state (https://github.com/pytorch/examples/blob/master/imagenet/main.py)
def save_checkpoint(state, is_best, filename='../checkpoints/checkpoint.pth.tar'):
    """
    :param state:
    :param is_best:
    :param filename:
    :return:
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '../checkpoints/model_best.pth.tar')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Road and Building Extraction')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset csv')
    parser.add_argument('--epochs', default=75, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='epoch to start from (used with resume flag')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--print-freq', default=4, type=int, metavar='N',
                        help='number of time to log per epoch')
    parser.add_argument('--run', default=0, type=int, metavar='N',
                        help='number of run (for tensorboard logging)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--data-set', default='mass_roads_crop', type=str,
                        help='mass_roads or mass_buildings or mass_roads_crop')

    args = parser.parse_args()

    main(args.data, batch_size=args.batch_size, num_epochs=args.epochs, start_epoch=args.start_epoch, learning_rate=args.lr,
         momentum=args.momentum, print_freq=args.print_freq, run=args.run, resume=args.resume, data_set=args.data_set)