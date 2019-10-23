"""
Utilization functions for creating subclasses.

History
-------
DATE       | DESCRIPTION    | NAME              | Organization |
21/07/2019 | first creation | Nikolaos Gkalelis | CERTH-ITI    |
"""

from torch.utils.data import DataLoader, sampler
import numpy as np
from sklearn.cluster import KMeans
import random
import torch

def getDatasetData(ds):
    """ partition_data: Partitions the selected classes to subclasses.
        :param ds: torchvision dataset
	"""

    #print("Getting dataset data")
    batchSize = 1000
    aLoader = DataLoader(ds, batch_size=batchSize, shuffle=False, pin_memory=True, num_workers=2)

    N = len(ds)
    x = np.zeros((N, ds.data.shape[1] * ds.data.shape[2] * ds.data.shape[3]))
    y = np.zeros((N,1))+ -1.0

    a = 0
    for batch_idx, (inputs, targets) in enumerate(aLoader):
        b = a + batchSize
        x_tmp = np.array(inputs)
        x[a:b, :] = x_tmp.flatten().reshape(x_tmp.shape[0], -1)
        y_tmp = np.array(targets)
        y[a:b, :] = y_tmp.flatten().reshape(y_tmp.shape[0], -1)
        a = b

    return x, y


def splitClass(x, y, ysb, ysb_incr, s, k):
    """ SplitClass: Split a class to subclasses.
        :param x: N x F array; N: number of observations, F: feature vector dimensionality
        :param y: N x 1 array; class labels, remain unaltered by this function
        :param ysb: array; subclass labels
        :param ysb_incr: array; subclass labels in incremental way
        :param s: scalar; label of the class to split
        :param k: number of subclasses to split class
    """


    print("Spliting class: {}".format(s))

    x1_tmp = x[y.flatten() == s, :] # get observations of class s
    x1 = x1_tmp.flatten().reshape(x1_tmp.shape[0], -1) # flatten returns a copy
    km1 = KMeans(n_clusters=k, random_state=0).fit(x1) # kmeans object
    ysb[y.flatten() == s, :] = km1.labels_[:, np.newaxis] # put the new subclass labels, 0 to k-1
    ll = np.max(ysb_incr) # label of last class

    for i in np.arange(0, k): # new class labels, one for each subclass: ll, ..., ll+k-1
        idxsub = np.logical_and(y.flatten() == s, ysb.flatten() == i)
        if i == 0:
            ysb_incr[idxsub, :] = s # the 1st subclass will keep the old class label
        else:
            ll = ll + 1
            ysb_incr[idxsub, :] = ll # new label

    C = int(ll + 1) # number of classes
    subclass2classIdx = np.zeros(C, dtype=np.int64)
    for i in np.arange(0, C):
        subclass2classIdx[i] = np.unique(y[ysb_incr == i]).item()

    return ysb, ysb_incr, subclass2classIdx

def augmentSelectedClasses(train_dataset, selcted_classes, dataset_name):
    """ augmentSelectedClasses: Augments selected classes.
        :param train_dataset: torchvision dataset
        :param selcted_classes: list with labels of selected classes
        :param num_subclasses_per_class: list with number of subclasses per class
        :param datasetName: name of the dataset
    """

    X = train_dataset.data

    if dataset_name == 'svhn':
        Y = train_dataset.labels
    else:
        Y = train_dataset.targets

    for c in selcted_classes:
        print("Augmenting class: {}".format(c))
        Y_np  = np.asarray(Y)
        X_c = X[Y_np == c, :, :, :] # class data
        Y_c_np = Y_np[Y_np  == c] # class targets
        #X_a = np.zeros((X_c.shape), dtype=X_c.dtype)
        N_c = X_c.shape[0] # class number of observations
        X_a = np.uint8((np.asarray(random.choices(X, k=N_c)) + X_c) / 2. ) # SamplePairing
        train_dataset.data = np.concatenate((train_dataset.data, X_a), axis=0) # augmentation

        if dataset_name == 'svhn':
            train_dataset.labels = np.concatenate((train_dataset.labels, Y_c_np))
        else:
            train_dataset.targets = train_dataset.targets + Y_c_np.tolist()

    return train_dataset

def partition_data(datasetName,
                   train_dataset,
                   selcted_classes,
                   num_subclasses_per_class,
                   ):
    """ partition_data: Partitions the selected classes to subclasses.
        :param datasetName: name of the dataset
        :param train_dataset: torchvision dataset
        :param selcted_classes: list with labels of selected classes
        :param num_subclasses_per_class: list with number of subclasses per class
	"""

    Nsc = len(selcted_classes)
    train_dataset = augmentSelectedClasses(train_dataset, selcted_classes, datasetName)
    x_train, y_train = getDatasetData(train_dataset) # get all dataset
    y_train_subclass = np.zeros(y_train.shape)  # initialize subclass index
    y_train_subclass_increm = np.copy(y_train)  # initialize incremental subclass index

    subclass2classIdx = []
    for i in np.arange(0, Nsc):
        y_train_subclass, y_train_subclass_increm, subclass2classIdx = splitClass(x_train, y_train,
                                                                                  y_train_subclass,
                                                                                  y_train_subclass_increm,
                                                                                  selcted_classes[i],
                                                                                  num_subclasses_per_class[i])
    num_classes = len(np.unique(subclass2classIdx))
    
    classSubclasses = [] # list of tuples with sublcass indices
    for i in np.arange(0, num_classes ):
        sbc, = np.where(subclass2classIdx == i) # subclasses of i-th class
        classSubclasses.insert(i,sbc)

    # transform targets to subclass labels and add training transformation
    if datasetName == 'cifar10' or datasetName == 'cifar100':
        train_dataset.targets = y_train_subclass_increm.flatten().astype(int).tolist().copy()
    elif datasetName == 'svhn':
        train_dataset.labels = y_train_subclass_increm.flatten().astype(train_dataset.labels.dtype).copy()
    else:
        raise NameError("Unexpected dataset name: " + datasetName)

    subclass2classIdx = torch.from_numpy(subclass2classIdx)

    return train_dataset, subclass2classIdx, classSubclasses
