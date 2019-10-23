"""
Implementation in PyTorch of the method and multiclass classification experiments 
in CIFAR10, CIFAR100 and SVHN datasets extending the VGG16 and the Wide Residual Network
architecture, as described in the paper:
N. Gkalelis, V. Mezaris, "Subclass deep neural networks: re-enabling neglected classes
in deep network training for multimedia classification", Proc. 26th Int. Conf. on
Multimedia Modeling (MMM2020), Daejeon, Korea, Jan. 2020.

History
-------
DATE       | DESCRIPTION    | NAME              | Organization |
21/07/2019 | first creation | Nikolaos Gkalelis | CERTH-ITI    |
"""

import argparse
import numpy as np
import json
from datetime import datetime
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import datasets, transforms

from util.cutout import Cutout
from util.subclassutils import partition_data
from util.trainutils import train_subclass_one_epoch, compute_neglected_classes
from util.testutils import test

from model.wide_resnet import WideResNet
from model.vgg import vgg16_bn

model_options = ['wideresnet', 'vgg16']
dataset_options = ['cifar10', 'cifar100', 'svhn']

parser = argparse.ArgumentParser(description='SCNN')
parser.add_argument('--dataset', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', default='vgg16',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--epochs_class', type=int, default=10,
                    help='number of epochs to train for identifying the neglected classes')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train the SDNN')
parser.add_argument('--learning_rate', type=float, default=0.1, # cifar10/100: 0.1; svhn: 0.01
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=True,
                    help='augment data by flipping and cropping') # cifar10/100: True; svhn: False
parser.add_argument('--cutout', action='store_true', default=False,  help='apply cutout')
parser.add_argument('--subclass', action='store_true', default=False, help='apply subclasses')

# 1. means ignore misclassification cost of assigning the observation to another subclass of same class
# 0. means place full misclassification cost of assigning the observation to another subclass of same class,
# i.e., the other subclasses are treated as completely different classes
parser.add_argument('--subclass_label_weight', type=float, default=0.9, help='Weight in [0,1] to weight the subclass label'
                         'of the subclasses belonging to different classes in the subclass CE criterion')

parser.add_argument('--numClasses2Repartition', default=2, type=int,
                    help='classes to repartition')
parser.add_argument('--numSubclassesPerClass', default=2, type=int,
                    help='number of subclasses for each class we partition')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')  # cifar10: 16; cifar100: 8; svhn: 20
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')

if __name__ == '__main__':

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True  # Should make training should go faster for large models

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print(args)

    # Normalization
    if args.dataset == 'svhn':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in[109.9, 109.7, 113.8]],
                                         std=[x / 255.0 for x in [50.1, 50.6, 50.8]])
    else:
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    # transforms
    train_transform = transforms.Compose([])
    if args.data_augmentation:
        train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
        train_transform.transforms.append(transforms.RandomHorizontalFlip())

    train_transform.transforms.append(transforms.ToTensor())
    train_transform.transforms.append(normalize)

    # cutout
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))

    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    # dataset
    if args.dataset == 'cifar10':
        num_classes = 10
        train_dataset = datasets.CIFAR10(root='data/',
                                         train=True,
                                         transform=train_transform,
                                         download=True)

        test_dataset = datasets.CIFAR10(root='data/',
                                        train=False,
                                        transform=test_transform,
                                        download=True)
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_dataset = datasets.CIFAR100(root='data/',
                                          train=True,
                                          transform=train_transform,
                                          download=True)

        test_dataset = datasets.CIFAR100(root='data/',
                                         train=False,
                                         transform=test_transform,
                                         download=True)
    elif args.dataset == 'svhn':
        num_classes = 10
        train_dataset = datasets.SVHN(root='data/',
                                      split='train',
                                      transform=train_transform,
                                      download=True)

        extra_dataset = datasets.SVHN(root='data/',
                                      split='extra',
                                      transform=train_transform,
                                      download=True)

        # Combine training and extra datasets
        data = np.concatenate([train_dataset.data, extra_dataset.data], axis=0)
        labels = np.concatenate([train_dataset.labels, extra_dataset.labels], axis=0)
        train_dataset.data = data
        train_dataset.labels = labels

        test_dataset = datasets.SVHN(root='data/',
                                     split='test',
                                     transform=test_transform,
                                     download=True)
    else:
        raise NameError("Unexpected dataset name: " + args.dataset)

    balancingSampler = None
    trnLdrDoShuffling = True

    ############################
    # identify neglected classes
    ############################
	
    train_loader_class = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=trnLdrDoShuffling,
                                               pin_memory=True,
                                               num_workers=2,
                                               sampler=balancingSampler)

    if args.model == 'wideresnet':
        if args.dataset == 'svhn':
            cnn_class = WideResNet(depth=16, num_classes=num_classes, widen_factor=8,
                             dropRate=0.4)

        else:
            cnn_class = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,
                             dropRate=0.3)
    elif args.model == 'vgg16':
        cnn_class = vgg16_bn(numClass=num_classes)
    else:
        raise NameError("Unexpected model name: " + args.model)

    cnn_class = cnn_class.cuda()
    cnn_class_optimizer = torch.optim.SGD(cnn_class.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True,
                                    weight_decay=5e-4)
    scheduler_class = MultiStepLR(cnn_class_optimizer, milestones=[10, 20, 30], gamma=0.2)
    cnn_optimizer_class = torch.optim.SGD(cnn_class.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True,
                                    weight_decay=5e-4)

    tti = compute_neglected_classes(train_loader_class, cnn_class, cnn_optimizer_class,
                                          scheduler_class, args.epochs_class, num_classes)

    args.class2repartition = torch.argsort(input=tti,  descending=True)[0:args.numClasses2Repartition].tolist()

    del cnn_class_optimizer, scheduler_class, cnn_class, train_loader_class

    ############
    # train scnn
    ############

    # augment neglected classes and subclass them
    numSubclassesPerClass = [args.numSubclassesPerClass] * args.numClasses2Repartition
    train_dataset, subclass2classIdx, classSubclasses = partition_data(args.dataset,
                                                      train_dataset,
                                                      args.class2repartition,
                                                      numSubclassesPerClass)
    subclass2classIdx = subclass2classIdx.cuda()
    num_subclasses = len(subclass2classIdx)

    # data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                     batch_size=args.batch_size,
                                                     shuffle=trnLdrDoShuffling,
                                                     pin_memory=True,
                                                     num_workers=2,
                                                     sampler=balancingSampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=2)

    # cnn architectures
    if args.model == 'wideresnet':
        if args.dataset == 'svhn':
            cnn = WideResNet(depth=16, num_classes=num_subclasses, widen_factor=8,
                             dropRate=0.4)

        else:
            cnn = WideResNet(depth=28, num_classes=num_subclasses, widen_factor=10,
                             dropRate=0.3)
    elif args.model == 'vgg16':
        cnn = vgg16_bn(numClass=num_subclasses)
    else:
        raise NameError("Unexpected model name: " + args.model)
	
    cnn = cnn.cuda()
    cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if args.dataset == 'svhn':
        scheduler = MultiStepLR(cnn_optimizer, milestones=[80, 120, 160, 220], gamma=0.1)
    else:
        scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160, 220, 260], gamma=0.2)

    test_id = args.dataset + '_' + args.model + '_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = 'logs/' + test_id + '.csv'
    best_model_filename = 'checkpoints/' + test_id + '.pt'
    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'trn_acc', 'tst_acc'
    ])

    # train the scnn
    best_acc = 0.
    for epoch in range(args.epochs):

        scheduler.step(epoch)
        train_acc = train_subclass_one_epoch(train_loader, cnn, cnn_optimizer, epoch, subclass2classIdx, classSubclasses, args.subclass_label_weight)
        test_acc, test_time = test(test_loader, cnn, subclass2classIdx)

        tmp = pd.Series([
            epoch,
            scheduler.get_lr()[0],
            train_acc,
            test_acc
        ], index=['epoch', 'lr', 'trn_acc', 'tst_acc'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv(log_filename, index=False)

        print('test_acc: {}, test_time: {}'.format(test_acc, test_time))

        if test_acc > best_acc:
            torch.save(cnn.state_dict(), best_model_filename)
            best_acc = test_acc
            print("=> saved best model")
