from model import cnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

EPOCH = 100

if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root='./data', train=True,
                                download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    testset = datasets.CIFAR10(root='./data', train=False,
                               download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)

    net = cnn(10, 3).to(device)
    criterion = nn.CrossEntropyLoss()  # 分类问题用交叉熵损失
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # 定义优化器（随机梯度下降）

    for epoch in range(EPOCH):
        correct = 0.0
        loss_tot = 0.0
        total_s = 0
        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)
            o, f = net(inputs)
            loss = criterion(o, labels)
            _, y_pred = o.data.max(1, keepdim=True)
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().sum().item()
            loss_tot += loss.item()
            total_s += len(inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_acc = correct / total_s
        avg_loss = loss_tot / total_s
        print(f"epoch:{epoch}, " + f"loss:{avg_loss}," + f"acc:{avg_acc}.")
    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in testloader:
            images, labels = images.to('cuda:0'), labels.to('cuda:0')
            outputs, _ = net(images)
            numbers, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
<<<<<<< HEAD
    torch.save(net.state_dict(), 'model.pth')
=======
>>>>>>> 77a4ebe (first update)
