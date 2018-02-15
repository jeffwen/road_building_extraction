from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

from logger import Logger
from src.utils import data_utils
from src.utils import augmentation as aug
from src.utils import metrics
from src.models import unet

import torch
import torch.optim as optim
import time
import argparse


def main(data_path, batch_size, num_epochs, learning_rate, momentum):
    """

    Args:
        data_path:
        batch_size:
        num_epochs:

    Returns:

    """
    since = time.time()

    # get data
    mass_dataset_train = data_utils.MassRoadBuildingDataset(data_path, 'mass_roads', 'train',
                                                       transform=transforms.Compose([aug.RescaleTarget((1000, 1400)),
                                                                         aug.RandomCropTarget(768),
                                                                         aug.ToTensorTarget(),
                                                                         aug.NormalizeTarget(mean=[0.5, 0.5, 0.5],
                                                                                             std=[0.5, 0.5, 0.5])]))

    mass_dataset_val = data_utils.MassRoadBuildingDataset(data_path, 'mass_roads', 'valid',
                                                     transform=transforms.Compose([aug.ToTensorTarget(),
                                                                         aug.NormalizeTarget(mean=[0.5, 0.5, 0.5],
                                                                                             std=[0.5, 0.5, 0.5])]))

    # creating loaders
    train_dataloader = DataLoader(mass_dataset_train, batch_size=batch_size, num_workers=6, shuffle=True)
    val_dataloader = DataLoader(mass_dataset_val, batch_size=6, num_workers=6, shuffle=False)

    # get model
    model = unet.UNet()

    if torch.cuda.is_available():
        model = model.cuda()

    # set up binary cross entropy and dice loss
    criterion = metrics.BCEDiceLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train(train_dataloader, model, criterion, optimizer, lr_scheduler)
        validation(val_dataloader, model, criterion)

        cur_elapsed = time.time() - since
        print('Current elapsed time {:.0f}m {:.0f}s'.format(cur_elapsed // 60, cur_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


def train(train_loader, model, criterion, optimizer, scheduler):
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

    scheduler.step()

    # Iterate over data.
    for data in train_loader:
        # get the inputs
        inputs = data['sat_img']
        labels = data['map_img']

        # wrap in Variable
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        prob_map = model(inputs) # last activation was a sigmoid
        outputs = (prob_map > 0.3).float()

        loss = criterion(outputs, labels)

        # backward
        loss.backward()
        optimizer.step()

        train_acc.update(metrics.dice_coeff(inputs, labels), inputs.size(0))
        train_loss.update(loss.data[0], inputs.size(0))

    print('Training Loss: {:.4f} Acc: {:.4f}'.format(train_loss.avg, train_acc.avg))
    print()


def validation(valid_loader, model, criterion):
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

    # switch to evaluate mode
    model.eval()

    # Iterate over data.
    for data in valid_loader:
        # get the inputs
        inputs = data['sat_img']
        labels = data['map_img']

        # wrap in Variable
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)
        else:
            inputs = Variable(inputs, volatile=True)
            labels = Variable(labels, volatile=True)

        # forward
        prob_map = model(inputs) # last activation was a sigmoid
        outputs = (prob_map > 0.3).float()

        loss = criterion(outputs, labels)

        valid_acc.update(metrics.dice_coeff(inputs, labels), inputs.size(0))
        valid_loss.update(loss.data[0], inputs.size(0))

    print('Validation Loss: {:.4f} Acc: {:.4f}'.format(valid_loss.avg, valid_acc.avg))
    print()

    return valid_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Road and Building Extraction')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()

    main(args.data, batch_size=args.batch_size, num_epochs=args.epochs, learning_rate=args.lr, momentum=args.momentum)