from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utils.config_utils import read_args, load_config


class Net(nn.Module):
    def __init__(self):#constructor
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x) #input
        x = F.relu(x)  #output
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch,rank):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    tot_loss = 0
    tot_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() #梯度清零
        output = model(data) #输入训练集
        loss = F.nll_loss(output, target) #计算损失函数
        loss.backward()
        optimizer.step()
        train_outputs = output.argmax(dim = 1)
        tot_loss += loss.data
        tot_acc += (train_outputs == target).sum().item()

        if batch_idx % args.log_interval == 0:
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
            print('Process:{}\t Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                rank, epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    training_acc = tot_acc / len(train_loader.dataset)
    training_loss = tot_loss / len(train_loader.dataset)
    return training_acc, training_loss




def tes(model, device, test_loader):
    #campare the result
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            prediction = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += prediction.eq(target.view_as(prediction)).sum().item()

        test_loss /= len(test_loader.dataset)
        testing_loss = test_loss
        testing_acc = correct/len(test_loader.dataset)
    return testing_acc, testing_loss




def run(config, rank):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()
    seed = [config.seed[0], config.seed[1], config.seed[2]]
    torch.manual_seed(seed[rank])
    print(str(rank)+"   " + str(seed[rank]))
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs,)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs,)


    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    #start training
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch, rank)
        """record training info, Fill your code"""
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        test_acc, test_loss = tes(model, device, test_loader)
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)
        scheduler.step()
    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    with open('../pythonProject4/resulttxt/trainLoss.txt', 'a') as tfile:
        tfile.write(str(training_loss))

    with open('../pythonProject4/resulttxt/trainAcc.txt', 'a') as file:
        file.write(str(training_accuracies))

    with open('../pythonProject4/resulttxt/testLoss.txt', 'a') as testfile:
        testfile.write(str(testing_loss))
    with open('../pythonProject4/resulttxt/testAcc.txt', 'a') as afile:
        afile.write(str(testing_accuracies))

    plot(training_loss, training_accuracies, testing_loss, testing_accuracies, config.epochs)


def plot(training_loss,training_accuracies,testing_loss,testing_accuracies,epochs):
    x = np.arange(0,epochs+1)
    training_loss.insert(0,1)
    y = np.array(training_loss)
    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.title('training loss')
    plt.plot(x,y,color = 'blue')
    plt.grid(True)
    plt.show()

    xs = np.arange(0,epochs+1)
    training_accuracies.insert(0,0)
    ys = np.array(training_accuracies)
    plt.xlabel('epochs')
    plt.ylabel('training accuracies')
    plt.title('training accuracy')
    plt.plot(xs,ys,color = 'red')
    plt.grid(True)
    plt.show()

    xp = np.arange(0,epochs+1)
    testing_loss.insert(0,1)
    yp = np.array(testing_loss)
    plt.xlabel('epochs')
    plt.ylabel('testing loss')
    plt.title('testing loss')
    plt.plot(xp,yp,color = 'blue')
    plt.grid(True)
    plt.show()

    xu = np.arange(0,epochs+1)
    testing_accuracies.insert(0,0)
    yu = np.array(testing_accuracies)
    plt.xlabel('epochs')
    plt.ylabel('testing accuracies')
    plt.title('testing accuracies')
    plt.grid(True)
    plt.plot(xu,yu,color = 'red')
    plt.show()
def plotMean():
    trainAcc = []
    meanTrainAcc = []
    with open('testAcc.txt', 'r') as file:
        for i in range(3):
            line = file.readline()
            trainAcc.append([float(i) for i in line.split(',')])
        print(trainAcc)
    sum = 0
    for j in range(15):
        for i in range(3):
            sum += trainAcc[i][j]
            a = sum / 3
        meanTrainAcc.append(a)
        sum = 0
        a = 0
    meanTrainAcc.insert(0,0)
    x = np.arange(0, 16)
    y = np.array(meanTrainAcc)
    plt.plot(x, y, color='red')
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('mean train acc')
    plt.title('mean train acc')
    plt.show()

    trainLoss = []
    meanTrainLoss = []
    with open('testLoss.txt', 'r') as file:
        for i in range(3):
            line = file.readline()
            trainLoss.append([float(i) for i in line.split(',')])
    sum = 0
    for j in range(15):
        for i in range(3):
            sum += trainLoss[i][j]
            a = sum / 3
        meanTrainLoss.append(a)
        sum = 0
        a = 0
    meanTrainLoss.insert(0, 1)
    x = np.arange(0, 16)
    y = np.array(meanTrainLoss)
    plt.plot(x, y, color='green')
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('mean train loss')
    plt.title('mean train loss')
    plt.show()

    testLoss = []
    meanTestLoss = []
    with open('testLoss.txt', 'r') as file:
        for i in range(3):
            line = file.readline()
            testLoss.append([float(i) for i in line.split(',')])
    sum = 0
    for j in range(15):
        for i in range(3):
            sum += testLoss[i][j]
            a = sum / 3
        meanTestLoss.append(a)
        sum = 0
        a = 0
    meanTestLoss.insert(0, 1)
    x = np.arange(0, 16)
    y = np.array(meanTestLoss)
    plt.plot(x, y, color='blue')
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('mean test loss')
    plt.title('mean test loss')
    plt.show()

    testAcc = []
    meanTestAcc = []
    with open('testAcc.txt', 'r') as file:
        for i in range(3):
            line = file.readline()
            testAcc.append([float(i) for i in line.split(',')])
        print(testAcc)
    sum = 0
    for j in range(15):
        for i in range(3):
            sum += testAcc[i][j]
            a = sum / 3
        meanTestAcc.append(a)
        sum = 0
        a = 0
    meanTestAcc.insert(0, 0)
    x = np.arange(0, 16)
    y = np.array(meanTestAcc)
    plt.plot(x, y, color='red')
    plt.grid(True)
    plt.xlabel('epochs')
    plt.ylabel('mean test acc')
    plt.title('mean test acc')
    plt.show()


if __name__ == '__main__':
    arg = read_args()
    config = load_config(arg)
    num_processes = 3
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = Net().to(device)
    model.share_memory()
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target = run, args = (config, rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    plotMean()

