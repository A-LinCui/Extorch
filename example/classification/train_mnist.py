"""
An example file for training a LeNet model on MNIST.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

import extorch.vision.dataset as dataset
import extorch.utils as utils
from extorch.nn import CrossEntropyMixupLoss 

from module import MNISTLeNet


def train_epoch(net, trainloader, device, optimizer, criterion, epoch, report_every, logger, grad_clip = None):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    net.train()
    for step, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = net(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, labels, topk = (1, 5))
        n = inputs.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        del loss
        if (step + 1) % report_every == 0:
            logger.info("Epoch {} train {} / {} {:.3f}; {:.3f}%; {:.3f}%".format(
                epoch, step + 1, len(trainloader), objs.avg, top1.avg, top5.avg))
   
    return objs.avg, top1.avg, top5.avg


def valid(net, testloader, device, optimizer, criterion, epoch, report_every, logger):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    net.eval()
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(testloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = net(inputs)
            loss = criterion(logits, labels)
            prec1, prec5 = utils.accuracy(logits, labels, topk = (1, 5))
            n = inputs.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            del loss
            if (step + 1) % report_every == 0:
                logger.info("Epoch {} valid {} / {} {:.3f}; {:.3f}%; {:.3f}%".format(
                    epoch, step + 1, len(testloader), objs.avg, top1.avg, top5.avg))
    
    return objs.avg, top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type = str, default = "./")
    parser.add_argument("--train-dir", type = str, default = None)
    parser.add_argument("--save-every", type = int, default = 50)
    parser.add_argument("--gpu", type = int, default = 0, help = "gpu device id")
    parser.add_argument("--epochs", default = 100, type = int)
    parser.add_argument("--num-workers", default = 4, type = int)
    parser.add_argument("--batch-size", default = 256, type = int)
    parser.add_argument("--report-every", default = 100, type = int)
    parser.add_argument("--seed", default = None, type = int)
    parser.add_argument("--lr", default = 0.01, type = float)
    parser.add_argument("--weight-decay", default = 1.e-4, type = float)
    parser.add_argument("--momentum", default = 0.9, type = float)
    parser.add_argument("--eta_min", default = 0., type = float)
    parser.add_argument("--grad-clip", default = 5., type = float)
    args = parser.parse_args()

    LOGGER = utils.getLogger("Main")

    if args.train_dir:
        utils.makedir(args.train_dir, remove = True)
        LOGGER.addFile(os.path.join(args.train_dir, "train.log"))
        writer = SummaryWriter(os.path.join(args.train_dir, "tensorboard"))

    DEVICE = torch.device("cuda:{}".format(args.gpu)) \
            if torch.cuda.is_available() else torch.device("cpu")

    if args.seed:
        utils.set_seed(args.seed)
        LOGGER.info("Set seed: {}".format(args.seed))

    # Use the MNIST dataset in extorch with the default transformation
    datasets = dataset.MNIST(args.data_dir)
    trainloader = data.DataLoader(dataset = datasets.splits["train"], \
            batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True)
    testloader = data.DataLoader(dataset = datasets.splits["test"], \
            batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)

    # Construct the network
    net = MNISTLeNet(num_classes = datasets.num_classes()).to(DEVICE)
    num_params = utils.get_params(net)
    LOGGER.info("Parameter size: {:.5f}M".format(num_params / 1.e6))
 
    criterion = CrossEntropyMixupLoss(alpha = 1.)

    # Construct the optimizer
    optimizer = optim.SGD(list(net.parameters()), lr = args.lr, 
                          weight_decay = args.weight_decay, momentum = args.momentum)

    # Construct the learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, eta_min = args.eta_min, T_max = args.epochs)
    
    time_estimator = utils.TimeEstimator(args.epochs)

    for epoch in range(1, args.epochs + 1):
        LOGGER.info("Epoch {} lr {:.5f}".format(epoch, optimizer.param_groups[0]["lr"]))

        loss, acc, acc_top5 = train_epoch(net, trainloader, DEVICE, optimizer, criterion, 
                epoch, args.report_every, LOGGER, args.grad_clip) 
        LOGGER.info("Train epoch {} / {}: obj {:.3f}; Acc. Top-1 {:.3f}%; Top-5 {:.3f}%".format(
                epoch, args.epochs, loss, acc, acc_top5))

        if args.train_dir:
            writer.add_scalar("Train/Loss", loss, epoch)
            writer.add_scalar("Train/Acc", acc / 100., epoch)
            writer.add_scalar("Train/Acc-top5", acc_top5 / 100., epoch)

        loss, acc, acc_top5 = valid(net, testloader, DEVICE, optimizer, criterion, 
                epoch, args.report_every, LOGGER) 
        LOGGER.info("TEST epoch {} / {}: obj {:.3f}; Acc. Top-1 {:.3f}%; Top-5 {:.3f}%".format(
                epoch, args.epochs, loss, acc, acc_top5))

        if args.train_dir:
            writer.add_scalar("Test/Loss", loss, epoch)
            writer.add_scalar("Test/Acc", acc / 100., epoch)
            writer.add_scalar("Test/Acc-top5", acc_top5 / 100., epoch)

        if epoch % args.save_every == 0 and args.train_dir:
            save_path = os.path.join(args.train_dir, "model_state_{}.ckpt".format(epoch))
            torch.save(net.state_dict(), save_path)
            LOGGER.info("Save checkpoint at {}".format(save_path))
        
        LOGGER.info("Iter {} / {} Remaining time: {} / {}".format(epoch, args.epochs, *time_estimator.step()))

        scheduler.step()

    if args.train_dir:
        save_path = os.path.join(args.train_dir, "final.ckpt")
        torch.save(net.state_dict(), save_path)
        LOGGER.info("Save checkpoint at {}".format(save_path))


if __name__ == "__main__":
    main()
