"""
An example file for adversarially training a ResNet-18 model on CIFAR-10.
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data as data

import extorch.vision.dataset as dataset
import extorch.utils as utils
import extorch.nn as extnn
from extorch.adversarial import PGDAdversary

from module import ResNet18


def train_epoch(net, trainloader, device, optimizer, criterion, adversary, epoch, 
        report_every, logger, grad_clip = None):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    adv_top1 = utils.AverageMeter()
    adv_top5 = utils.AverageMeter()
    
    net.train()

    for step, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        optimizer.zero_grad()

        with torch.no_grad():
            logits = net(inputs)
            
        adv_inputs = adversary(net, inputs, labels, logits)
        adv_logits = net(adv_inputs)

        loss = criterion(adv_logits, labels)
        loss.backward()

        if grad_clip is not None:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, labels, topk = (1, 5))
        adv_prec1, adv_prec5 = utils.accuracy(adv_logits, labels, topk = (1, 5))
            
        n = inputs.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        adv_top1.update(adv_prec1.item(), n)
        adv_top5.update(adv_prec5.item(), n)

        del loss

        if (step + 1) % report_every == 0:
            logger.info("Epoch {} train {} / {} {:.3f}; {:.3f}%; {:.3f}%; {:.3f}%; {:.3f}%".format(
                epoch, step + 1, len(trainloader), objs.avg, top1.avg, top5.avg, 
                adv_top1.avg, adv_top5.avg))
   
    return objs.avg, top1.avg, adv_top1.avg


def valid(net, testloader, device, optimizer, criterion, adversary, epoch, report_every, logger):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    adv_top1 = utils.AverageMeter()
    adv_top5 = utils.AverageMeter()

    net.eval()
    for step, (inputs, labels) in enumerate(testloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
            
        with torch.no_grad():
            logits = net(inputs)
            
        adv_inputs = adversary(net, inputs, labels, logits)
        adv_logits = net(adv_inputs)

        loss = criterion(adv_logits, labels)
                
        prec1, prec5 = utils.accuracy(logits, labels, topk = (1, 5))
        adv_prec1, adv_prec5 = utils.accuracy(adv_logits, labels, topk = (1, 5))
        
        n = inputs.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        adv_top1.update(adv_prec1.item(), n)
        adv_top5.update(adv_prec5.item(), n)

        del loss

        if (step + 1) % report_every == 0:
            logger.info("Epoch {} valid {} / {} {:.3f}; {:.3f}%; {:.3f}%; {:.3f}%; {:.3f}".format(
                epoch, step + 1, len(testloader), objs.avg, top1.avg, top5.avg, 
                adv_top1.avg, adv_top5.avg))
    
    return objs.avg, top1.avg, adv_top1.avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type = str, default = "./")
    parser.add_argument("--train-dir", type = str, default = None)
    parser.add_argument("--save-every", type = int, default = 50)
    parser.add_argument("--gpu", type = int, default = 0, help = "gpu device id")
    parser.add_argument("--epochs", default = 110, type = int)
    parser.add_argument("--num-workers", default = 4, type = int)
    parser.add_argument("--batch-size", default = 48, type = int)
    parser.add_argument("--report-every", default = 100, type = int)
    parser.add_argument("--seed", default = None, type = int)
    parser.add_argument("--epsilon", default = 0.2, type = float, help = "label smooth coefficient")
    parser.add_argument("--lr", default = 0.05, type = float)
    parser.add_argument("--weight-decay", default = 5e-4, type = float)
    parser.add_argument("--momentum", default = 0.9, type = float)
    parser.add_argument("--gamma", default = 0.1, type = float)
    parser.add_argument("--grad-clip", default = 5., type = float)
    parser.add_argument("--milestones", nargs = "*", default = [99, 104])
    parser.add_argument("--pgd-epsilon", default = 8. / 255, type = float)
    parser.add_argument("--n-step", default = 7, type = int)
    parser.add_argument("--step-size", default = 2. / 255, type = float)
    parser.add_argument("--use-eval-mode", action = "store_true", default = True)
    parser.add_argument("--rand-init", action = "store_true", default = True, 
            help = "random init before running PGD attack")
    args = parser.parse_args()

    LOGGER = utils.getLogger("Main")

    if args.train_dir:
        utils.makedir(args.train_dir, remove = True)
        LOGGER.addFile(os.path.join(args.train_dir, "train.log"))

    DEVICE = torch.device("cuda:{}".format(args.gpu)) \
            if torch.cuda.is_available() else torch.device("cpu")

    if args.seed:
        utils.set_seed(args.seed)
        LOGGER.info("Set seed: {}".format(args.seed))

    # Use the CIFAR-10 dataset in extorch with the default transformation
    datasets = dataset.CIFAR10(args.data_dir)
    trainloader = data.DataLoader(dataset = datasets.splits()["train"], \
            batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True)
    testloader = data.DataLoader(dataset = datasets.splits()["test"], \
            batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)

    # Construct the network
    net = ResNet18(num_classes = datasets.num_classes).to(DEVICE)
    num_params = utils.get_params(net)
    LOGGER.info("Parameter size: {:.5f}M".format(num_params / 1.e6))
 
    # Use the CrossEntropyLoss with label smooth in extorch
    criterion = extnn.CrossEntropyLabelSmooth(epsilon = args.epsilon)

    # Construct the optimizer
    optimizer = optim.SGD(list(net.parameters()), lr = args.lr, 
                          weight_decay = args.weight_decay, momentum = args.momentum)

    # Construct the learning rate scheduler
    scheduler = MultiStepLR(optimizer, gamma = args.gamma, milestones = args.milestones)

    # Construct the adversary
    adversary = PGDAdversary(epsilon = args.pgd_epsilon,
                             n_step = args.n_step,
                             step_size = args.step_size,
                             rand_init = args.rand_init,
                             use_eval_mode = args.use_eval_mode,
                             mean = dataset.cifar.CIFAR10_MEAN,
                             std = dataset.cifar.CIFAR10_STD)

    time_estimator = utils.TimeEstimator(args.epochs)

    for epoch in range(1, args.epochs + 1):
        loss, acc, acc_adv = train_epoch(net, trainloader, DEVICE, optimizer, criterion, 
                adversary, epoch, args.report_every, LOGGER, args.grad_clip) 
        LOGGER.info("Train epoch {} / {}: obj {:.3f}; acc {:.3f}%; adv_acc {:.3f}%".format(
                epoch, args.epochs, loss, acc, acc_adv))

        loss, acc, acc_adv = valid(net, testloader, DEVICE, optimizer, criterion, 
                adversary, epoch, args.report_every, LOGGER) 
        LOGGER.info("TEST epoch {} / {}: obj {:.3f}; acc {:.3f}%; adv_acc {:.3f}%".format(
                epoch, args.epochs, loss, acc, acc_adv))

        if epoch % args.save_every == 0 and args.train_dir:
            save_path = os.path.join(args.train_dir, "model_state_{}.ckpt".format(epoch))
            torch.save(net.state_dict(), save_path)
            LOGGER.info("Save checkpoint at {}".format(save_path))
        
        LOGGER.info("Iter {} / {} Remaining time: {} / {}".format(epoch, args.epochs, *time_estimator.step()))

    if args.train_dir:
        save_path = os.path.join(args.train_dir, "final.ckpt")
        torch.save(net.state_dict(), save_path)
        LOGGER.info("Save checkpoint at {}".format(save_path))


if __name__ == "__main__":
    main()
