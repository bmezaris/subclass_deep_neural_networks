"""
Implementation of the subclass sigmoid cross entropy loss.

History
-------
DATE       | DESCRIPTION    | NAME              | ORGANIZATION |
21/07/2019 | first creation | Nikolaos Gkalelis | CERTH-ITI    |
"""

import torch

def ClassSigmoidCrossEntropyLoss(predictions_sigmoid, float_labels_class_one_hot ):
    """ ClassSigmoidCrossEntropyLoss: Sigmoid cross-entropy loss.
        :param predictions_sigmoid: N x S array of predictions after sigmoid is applied; N: number of training observations, S: number of subclasses
        :param float_labels_class_one_hot: N x S float array with one hot class vector at each row
        :return: cross entropy loss: scalar value
    """

    epsilon = 10e-6
    onesMat = 1.
    cross_entropy_loss = float_labels_class_one_hot * torch.log(predictions_sigmoid + epsilon) + (
            onesMat - float_labels_class_one_hot) * torch.log( onesMat - predictions_sigmoid + epsilon)
    cross_entropy_loss = - cross_entropy_loss
    return torch.mean(torch.sum(cross_entropy_loss, dim=1))

def SubclassSigmoidCrossEntropyLoss(predictions, labels_subclass_one_hot, labels_class_many_hot):
    """ SubclassSigmoidCrossEntropyLoss: Subclass sigmoid cross-entropy loss.
        :param predictions: N x S array of predictions; N: number of training observations, S: number of subclasses
        :param labels_subclass_one_hot: N x S array with one hot subclass vector at each row
        :param labels_class_many_hot: N x S array with "many hot" subclass vector at each row (placing 1 to all subclasses that this observation is related to)
        :return: cross entropy loss: scalar value
    """

    epsilon = 10e-6
    float_labels_subclass_one_hot = labels_subclass_one_hot.float()
    float_labels_class_many_hot = labels_class_many_hot.float()

    predictions_sigmoid = torch.sigmoid(predictions)
    onesMat = 1.
    cross_entropy_loss = float_labels_subclass_one_hot * torch.log(predictions_sigmoid + epsilon) + (
            onesMat - float_labels_class_many_hot) * torch.log( onesMat - predictions_sigmoid + epsilon)
    cross_entropy_loss = - cross_entropy_loss
    return torch.mean(torch.sum(cross_entropy_loss, dim=1))
