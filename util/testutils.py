"""
Utilization functions for testing the SDNN.

History
-------
DATE       | DESCRIPTION    | NAME              | Organization |
21/07/2019 | first creation | Nikolaos Gkalelis | CERTH-ITI    |
"""

import torch, time

def test(loader, dnn, subclass2classIdx):
    """ train_subclass_one_epoch: Performs one epoch for training the subclass DNN.
        :param loader: torch DataLoader object
        :param dnn: deep neural network object
        :param subclass2classIdx: weight of the subclass label for subclasses belonging to different classes
	"""

    start_time = time.time()
    dnn.eval()  # Change model to 'eval' mode (do not update weights).
    correct = 0.
    total = 0.

    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = dnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)

        pred_class = subclass2classIdx[pred]
        correct += (pred_class == labels).sum().item()

    val_acc = correct / total

    test_time = time.time() - start_time
    # print("Test time: {} secs".format(test_time))

    return val_acc, test_time