import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time

import numpy as np
import matplotlib.pyplot as plt
import os


# prepare data
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

dataset = 'cifar-10-subset-old'
train_directory = os.path.join(dataset, 'train')
valid_directory = os.path.join(dataset, 'valid')
test_directory = os.path.join(dataset, 'test')

#######################################################
# change when modifying model
batch_size = 20
#######################################################
num_classes = 10

data = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['valid'])
}

train_data_size = len(data['train'])
valid_data_size = len(data['valid'])
test_data_size = len(data['test'])

train_data = DataLoader(data['train'], batch_size=batch_size, shuffle=True)
valid_data = DataLoader(data['valid'], batch_size=batch_size, shuffle=True)
test_data = DataLoader(data['test'], batch_size=batch_size, shuffle=True)

print(train_data_size, valid_data_size, test_data_size)


def train_and_valid(model, trainedModelDir, train_data, train_data_size, valid_data, valid_data_size, loss_function, optimizer, scheduler,
                    start_epoch=0, epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if start_epoch == 0:
        history = []
    else:
        history = torch.load(os.path.join(trainedModelDir, dataset+'_history.pth'))
    best_acc = 0.0
    best_epoch = 0

    try:
        for epoch in range(start_epoch, start_epoch+epochs):
            epoch_start = time.time()
            print("Epoch: {}/{}".format(epoch + 1, start_epoch+epochs))

            model.train()

            train_loss = 0.0
            train_acc = 0.0
            valid_loss = 0.0
            valid_acc = 0.0

            for i, (inputs, labels) in enumerate(train_data):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 因为这里梯度是累加的，所以每次记得清零
                optimizer.zero_grad()

                outputs = model(inputs)

                loss = loss_function(outputs, labels)

                loss.backward()

                optimizer.step()

                train_loss += loss.item() * inputs.size(0)

                ret, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))

                acc = torch.mean(correct_counts.type(torch.FloatTensor))

                train_acc += acc.item() * inputs.size(0)

            with torch.no_grad():
                model.eval()

                for j, (inputs, labels) in enumerate(valid_data):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)

                    loss = loss_function(outputs, labels)

                    valid_loss += loss.item() * inputs.size(0)

                    ret, predictions = torch.max(outputs.data, 1)
                    correct_counts = predictions.eq(labels.data.view_as(predictions))

                    acc = torch.mean(correct_counts.type(torch.FloatTensor))

                    valid_acc += acc.item() * inputs.size(0)

            avg_train_loss = train_loss / train_data_size
            avg_train_acc = train_acc / train_data_size

            avg_valid_loss = valid_loss / valid_data_size
            avg_valid_acc = valid_acc / valid_data_size

            history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])

            if best_acc < avg_valid_acc:
                best_acc = avg_valid_acc
                best_epoch = epoch + 1

            scheduler.step()

            epoch_end = time.time()

            print(
                "Epoch: {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation: Loss: {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(
                    epoch + 1, avg_train_loss, avg_train_acc * 100, avg_valid_loss, avg_valid_acc * 100,
                    epoch_end - epoch_start
                ))
            print("Best Accuracy for validation : {:.4f}% at epoch {:03d}".format(best_acc * 100, best_epoch))
            checkpoint = {'model':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': epoch + 1}
            torch.save(checkpoint, os.path.join(trainedModelDir, dataset + '_model_' + str(epoch + 1) + '.pth'))
    except KeyboardInterrupt:
        print("Best Accuracy for validation : {:.4f}% at epoch {:03d}".format(best_acc * 100, best_epoch))
    finally:
        if os.path.exists(os.path.join(trainedModelDir, dataset + '_history.pth')):
            os.remove(os.path.join(trainedModelDir, dataset + '_history.pth'))
        torch.save(history, os.path.join(trainedModelDir, dataset + '_history.pth'))
    return model, history


def test(model, test_data, test_data_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_acc = 0.0
    with torch.no_grad():
        model.eval()

        for j, (inputs, labels) in enumerate(test_data):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            ret, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))

            acc = torch.mean(correct_counts.type(torch.FloatTensor))

            test_acc += acc.item() * inputs.size(0)

    avg_test_acc = test_acc / test_data_size

    print("Testing: Accuracy: {:.4f}%".format(
        avg_test_acc * 100
    ))


def find_start_epoch(path):
    """
    :param path: path containing trained models
    :return: maximum trained history epoch number
    """
    if not os.path.exists(path):
        return 0
    import re
    fileList = os.listdir(path)
    ttt = []
    for st in fileList:
        match = re.search('\\d+.pt[h]?', st)
        if match:
            ttt.append(int(match.group(0).split('.')[0]))
    if len(ttt) == 0:
        return 0
    else:
        ttt = np.array(ttt)
        return ttt.max()


def main(NEW, num_epochs):
    """
    :param NEW: means whether we use fresh model or trained model for training, True means fresh model
    :param num_epochs: how many epochs will be performed on training the model
    :return:
    """
    ######################################################
    # model definition
    # needs string replace when changing model

    modelName = 'resnet34'
    trainedModelDir = 'models-%s' % modelName
    if not os.path.exists(trainedModelDir):
        os.mkdir(trainedModelDir)

    resnet34 = models.resnet34(pretrained=True)

    # for param in resnet34.parameters():
    #     param.requires_grad = False

    fc_inputs = resnet34.fc.in_features
    resnet34.fc = nn.Sequential(
        nn.Linear(fc_inputs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 10),
        nn.LogSoftmax(dim=1)
    )

    model = resnet34.to('cuda:0')
    loss_func = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    ############################################################
    ############################################################
    # change when starting another training
    # means whether we use fresh model or trained model for training, True means fresh model
    ############################################################
    if NEW:
        start_epoch = 0
        torch.save(model, os.path.join(trainedModelDir, dataset+'_model_%d.pth' % start_epoch))
    else:
        # epoch starts from 0
        start_epoch = find_start_epoch(trainedModelDir)
        checkpoint = torch.load(os.path.join(trainedModelDir, dataset+'_model_%d.pth' % start_epoch))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 35], gamma=0.1, last_epoch=start_epoch-1)
    # train and validation
    trained_model, history = train_and_valid(model, trainedModelDir, train_data, train_data_size,
                                             valid_data, valid_data_size,
                                             loss_func, optimizer, scheduler, start_epoch, num_epochs)


    epoch = np.arange(1, start_epoch+num_epochs+1, 1)
    history = np.array(history)
    plt.plot(epoch, history[:, 0:2])
    plt.legend(['Tr Loss', 'Val Loss'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.axis([0, 1.05 * (start_epoch+num_epochs), 0, min(1.1*history[:, 0:2].max(), 3)])
    plt.savefig(dataset+'_'+modelName+'_loss_curve_Epochs_%d.png' % (start_epoch + num_epochs))
    plt.show()

    plt.plot(epoch, history[:, 2:4])
    plt.legend(['Tr Accuracy', 'Val Accuracy'])
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.axis([0, 1.05 * (start_epoch+num_epochs), 0, 1])
    plt.savefig(dataset+'_'+modelName+'_accuracy_curve_Epochs_%d.png' % (start_epoch + num_epochs))
    plt.show()

    # test the model which has best valid accuracy
    valid_acc = history[:, 3]
    best_model_idx = valid_acc.argmax()
    print('Best performance on valid set at epoch %d with accuracy %.2f%%' % (best_model_idx + 1, 100*valid_acc[best_model_idx]))
    checkpoint = torch.load(os.path.join(trainedModelDir, dataset+'_model_'+str(best_model_idx+1)+'.pth'))
    # test
    model.load_state_dict(checkpoint['model'])
    test(model, test_data, test_data_size)


if __name__ == "__main__":
    try:
        main(True, 40)
    except Exception as err:
        print(err)
    finally:
        exit()
