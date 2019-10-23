"""
Utilization functions for training the SDNN.

History
-------
DATE       | DESCRIPTION    | NAME              | Organization |
21/07/2019 | first creation | Nikolaos Gkalelis | CERTH-ITI    |
"""

from tqdm import tqdm
from torch.nn.functional import one_hot
import torch
import pandas as pd

from util.mylosses import ClassSigmoidCrossEntropyLoss, SubclassSigmoidCrossEntropyLoss
from util.testutils import test

def compute_neglected_classes(train_loader,
                              dnn,
                              dnn_optimizer,
                              scheduler,
                              num_epochs,
                              num_classes):

    """ SplitClass: Splits the selected classes to subclasses.
        :param train_loader: torch DataLoader object
        :param dnn: deep neural network object
        :param dnn_optimizer: torch optimization object
        :param scheduler: torch optimization scheduler object
        :param num_epochs: number of epochs to run in order to identiyf the neglected classes
        :param num_classes: number of classes
	"""

    tti = torch.zeros([num_classes,], dtype=torch.float) # initialize neglection degrees (for each class)

    X = None

    def penultim_layer_hook(module, input_, output):
        nonlocal X
        X = output

    dnn.penultim_layer.register_forward_hook(penultim_layer_hook)

    dnn.train()

    for epoch in range(num_epochs):
        ti = torch.zeros([num_classes,], dtype=torch.float) # zero every epoch
        scheduler.step(epoch)

        num_batch = 0
        for num_batch, (images, labels) in enumerate(train_loader):

            images = images.cuda()
            labels = labels.cuda()

            pred_linear = dnn(images) # linear output of penultimate layer

            Ni = list(X.size())[0] # batch size

            float_labels_class_one_hot = one_hot(labels, num_classes=num_classes).float()
            pred_sigmoid  = torch.sigmoid(pred_linear)
            xentropy_loss = ClassSigmoidCrossEntropyLoss(pred_sigmoid , float_labels_class_one_hot).cuda()

            # compute gradient and do optimizing step
            dnn_optimizer.zero_grad()
            xentropy_loss.backward()
            dnn_optimizer.step()

            #dnn.eval()

            with torch.no_grad():
                Z = pred_sigmoid - float_labels_class_one_hot

                for i in labels:  # update overall class mean/cardinality vectors; only if class observations exist in this batch
                    Ri =  float_labels_class_one_hot[:, i] # indicator matrix for i-th class
                    X1 = X[Ri == 1, :]  # take positive observations of i-th class
                    Z1 = Z[Ri == 1, i]  # take gradient weights of positive observations of i-th class
                    X0 = X[Ri == 0, :]  # take negative observations of i-th class
                    Z0 = Z[Ri == 0, i]  # take gradient weights of negative observations of i-th class

                    g1i = torch.mm(Z1.unsqueeze(0), X1)
                    g0i = torch.mm(Z0.unsqueeze(0), X0)
                    gi = g1i + g0i
                    ti[i]  += (g1i.squeeze(0).norm() + g0i.squeeze(0).norm()) / gi.squeeze(0).norm()

            ti += ti / Ni

        #tti += ti / ( num_batch * num_epochs)
        tti = ti / num_batch
		
        #print('epoch: {}, ti: {}, tti: {}'.format(epoch, ti, tti))

    return tti

def train_subclass_one_epoch(train_loader,
                    dnn,
                    dnn_optimizer,
                    epoch,
                    subclass2classIdx,
                    classSubclasses,
                    subclassLabelWeight):
    """ train_subclass_one_epoch: Performs one epoch for training the subclass DNN.
        :param train_loader: torch DataLoader object
        :param dnn: deep neural network object
        :param dnn_optimizer: torch optimization object
        :param epoch: current epoch
        :param subclass2classIdx: converts subclass index to class index
        :param classSubclasses: list of tuples with sublcass indices
        :param subclassLabelWeight: weight of the subclass label for subclasses belonging to different classes
    """

    train_acc = 0.

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    dnn.train()
    num_subclasses = len(subclass2classIdx)

    progress_bar = tqdm(train_loader)
    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        pred = dnn(images)

        Nb = list(labels.size())  # batch size
        Nb = Nb[0]
        class_labels = subclass2classIdx[labels]
        labels_subclass_one_hot = one_hot(labels, num_classes=num_subclasses)

        wsc = subclassLabelWeight  # (1- wsc): weight for weighing loss of misclassifying to a subclass of the same class
        labels_class_many_hot = torch.zeros([Nb, num_subclasses], dtype=torch.float)
        for i in range(Nb):
            labels_class_many_hot[i, classSubclasses[class_labels[i]]] = wsc
            labels_class_many_hot[i, labels[i]] = 1.  # for the true subclass have weight always 1

        labels_class_many_hot = labels_class_many_hot.cuda()
        xentropy_loss = SubclassSigmoidCrossEntropyLoss(pred, labels_subclass_one_hot, labels_class_many_hot).cuda()

        # compute gradient and do optimizing step
        dnn_optimizer.zero_grad()
        xentropy_loss.backward()
        dnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        train_acc = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % train_acc)

    return train_acc