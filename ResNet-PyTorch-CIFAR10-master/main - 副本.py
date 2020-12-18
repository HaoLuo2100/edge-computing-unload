import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

# from model import ResNet
from torchvision import models

import numpy as np
import os
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='./dataset', type=str,
                    help='path to dataset')
parser.add_argument('--weight-decay', default=0.0001, type=float,
                    help='parameter to decay weights')
parser.add_argument('--batch-size', default=128, type=int,
                    help='size of each batch of cifar-10 training images')
parser.add_argument('--print-every', default=100, type=int,
                    help='number of iterations to wait before printing')
parser.add_argument('-n', default=5, type=int,
                    help='value of n to use for resnet configuration (see https://arxiv.org/pdf/1512.03385.pdf for details)')
parser.add_argument('--use-dropout', default=False, const=True, nargs='?',
                    help='whether to use dropout in network')
parser.add_argument('--res-option', default='A', type=str,
                    help='which projection method to use for changing number of channels in residual connections')

def main(args):
    # define transforms for normalization and data augmentation
    transform_augment = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(32, padding=4)])
    transform_normalize = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # get CIFAR-10 data
    dataset = 'cifar-10'
    NUM_TRAIN = 45000
    NUM_VAL = 5000
    cifar10_train = dset.CIFAR10('./dataset', train=True, download=False,
                                 transform=T.Compose([transform_augment, transform_normalize]))
    loader_train = DataLoader(cifar10_train, batch_size=args.batch_size,
                              sampler=ChunkSampler(NUM_TRAIN))
    cifar10_val = dset.CIFAR10('./dataset', train=True, download=False,
                               transform=transform_normalize)
    loader_val = DataLoader(cifar10_train, batch_size=args.batch_size,
                            sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN))
    cifar10_test = dset.CIFAR10('./dataset', train=False, download=False,
                                transform=transform_normalize)
    loader_test = DataLoader(cifar10_test, batch_size=args.batch_size)
    
    # load model
    # model = ResNet(args.n, res_option=args.res_option, use_dropout=args.use_dropout)
    model = models.resnet34(pretrained=True)
    modelName = 'resnet34'
    trainedModelDir = 'models-%s' % modelName
    if not os.path.exists(trainedModelDir):
        os.mkdir(trainedModelDir)

    for param in model.parameters():
        param.requires_grad = False

    fc_inputs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10),
        nn.LogSoftmax(dim=1)
    )

    param_count = get_param_count(model)
    print('Parameter count: %d' % param_count)
    
    # use gpu for training
    if not torch.cuda.is_available():
        print('Error: CUDA library unavailable on system')
        return
    global gpu_dtype
    gpu_dtype = torch.cuda.FloatTensor
    model = model.type(gpu_dtype)
    
    # setup loss function
    criterion = nn.CrossEntropyLoss().cuda()
    # train model
    SCHEDULE_EPOCHS = [20, 5, 5] # divide lr by 10 after each number of epochs
    total_epochs = sum(SCHEDULE_EPOCHS)
#     SCHEDULE_EPOCHS = [100, 50, 50] # divide lr by 10 after each number of epochs
    history = np.zeros((total_epochs, 3))    # train_loss, train_acc, valid_acc
    learning_rate = 0.1
    start_epoch = 0
    for i, num_epochs in enumerate(SCHEDULE_EPOCHS):
        if i >0:
            start_epoch += SCHEDULE_EPOCHS[i - 1]
        print('Training for %d epochs with learning rate %f' % (num_epochs, learning_rate))
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=0.9, weight_decay=args.weight_decay)
        for epoch in range(num_epochs):
            print('On Validation Set ', end='')
            valid_acc = check_accuracy(model, loader_val)
            print('Starting epoch %d / %d' % (epoch+1, num_epochs))
            train_loss = train(loader_train, model, criterion, optimizer)
            print('On Training Set ', end='')
            train_acc = check_accuracy(model, loader_train)
            history[start_epoch+epoch, :] = np.array([train_loss, train_acc, valid_acc])
            state_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': start_epoch+epoch+1}
            torch.save(state_dict, os.path.join(trainedModelDir, dataset+'_model_%d.pth' % (start_epoch+epoch+1)))

        learning_rate *= 0.1

    history.tofile(os.path.join(trainedModelDir, dataset+'_history.np'))

    best_model_idx = history[:, 2].argmax()
    checkpoint = torch.load(os.path.join(trainedModelDir, dataset+'_model_%d.pth' % (best_model_idx+1)))
    model.load_state_dict(checkpoint['model'])
    print('Final test accuracy:')
    check_accuracy(model, loader_test)
    # plots
    xidx = np.arange(1, total_epochs+1)
    plt.plot(xidx, history[:, 0])
    plt.legend(['Tr Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.axis([0, 1.05 * total_epochs, 0, 1.1 * history[:, 0].max()])
    plt.savefig(dataset + '_' + modelName + '_loss_curve_Epochs_%d.png' % total_epochs)
    plt.show()

    plt.plot(xidx, history[:, 1:3])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.axis([0, 1.05 * total_epochs, 0, 1.1])
    plt.savefig(dataset + '_' + modelName + '_accuracy_curve_Epochs_%d.png' % total_epochs)
    plt.show()


def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X_var = Variable(X.type(gpu_dtype))

        scores = model(X_var)
        _, preds = scores.data.cpu().max(1)

        num_correct += (preds == y).sum()
        num_samples += preds.size(0)

    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))
    return acc

def train(loader_train, model, criterion, optimizer):
    model.train()
    train_loss = 0
    train_data_size = loader_train.__sizeof__()[0]
    for t, (X, y) in enumerate(loader_train):
        X_var = Variable(X.type(gpu_dtype))
        y_var = Variable(y.type(gpu_dtype)).long()

        scores = model(X_var)

        loss = criterion(scores, y_var)
        train_loss += loss.item()

        if (t+1) % args.print_every == 0:
            print('t = %d, loss = %.4f' % (t+1, loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / train_data_size
    return avg_train_loss


def get_param_count(model):
    param_counts = [np.prod(p.size()) for p in model.parameters()]
    return sum(param_counts)

class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start
    
    def __iter__(self):
        return iter(range(self.start, self.start+self.num_samples))
    
    def __len__(self):
        return self.num_samples

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
