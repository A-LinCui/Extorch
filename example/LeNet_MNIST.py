"""
An example file for training a LeNet model on the FashionMNIST dataset with extorch API.
"""

import argparse

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data as data

import extorch.vision.dataset as dataset
import extorch.utils as utils
import extorch.nn as extnn


class LeNet(nn.Module):
    def __init__(self, num_classes: int):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

EPOCHS = 300
BATCH_SIZE = 256
NUM_WORKERS = 2

OPTIMIZER_KWARGS = {
        "lr": 3.e-3,
        "weight_decay": 1e-4,
        "momentum": 0.9
}

SCHEDULER_KWARGS = {
        "eta_min": 0.,
        "T_max": EPOCHS
}

REPORT_EVERY = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type = str, default = "./")
    args = parser.parse_args()

    LOGGER = utils.getLogger("Main")

    # Use the MNIST dataset in extorch with the default transformation
    datasets = dataset.MNIST(args.data_dir)
    trainloader = data.DataLoader(dataset = datasets.splits()["train"], \
            batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = True)
    testloader = data.DataLoader(dataset = datasets.splits()["test"], \
            batch_size = BATCH_SIZE, num_workers = NUM_WORKERS, shuffle = False)

    # Construct the network
    net = LeNet(num_classes = datasets.num_classes).to(DEVICE)
    num_params = utils.get_params(net)
    LOGGER.info("Parameter size: {:.5f}M".format(num_params / 1.e6))

    # Use the CrossEntropyLoss with label smooth in extorch
    criterion = extnn.CrossEntropyLabelSmooth(epsilon = 0.)

    # Construct the optimizer
    optimizer = optim.SGD(list(net.parameters()), **OPTIMIZER_KWARGS)

    # Construct the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, **SCHEDULER_KWARGS)

    # Start training
    for epoch in range(1, EPOCHS + 1):
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()

        net.train()
        for step, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()

            logits = net(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            prec1, prec5 = utils.accuracy(logits, labels, topk = (1, 5))
            n = inputs.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            del loss

            if (step + 1) % REPORT_EVERY == 0:
                LOGGER.info("epoch {} train {} {:.3f}; {:.3f}%; {:.3f}%".format(
                    epoch, step + 1, objs.avg, top1.avg, top5.avg))
            
        LOGGER.info("epoch {} train_obj {:.3f}; train_acc {:.3f}%".format(epoch, objs.avg, top1.avg))
        
        objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        net.eval()
        with torch.no_grad():
            for step, (inputs, labels) in enumerate(testloader):
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
            
                logits = net(inputs)
                loss = criterion(logits, labels)
                
                prec1, prec5 = utils.accuracy(logits, labels, topk = (1, 5))
                n = inputs.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                del loss

                if (step + 1) % REPORT_EVERY == 0:
                    LOGGER.info("epoch {} test {} {:.3f}; {:.3f}%; {:.3f}%".format(
                        epoch, step + 1, objs.avg, top1.avg, top5.avg))
            
            LOGGER.info("epoch {} test_obj {:.3f}; test_acc {:.3f}%".format(epoch, objs.avg, top1.avg))
