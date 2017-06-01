from copy import deepcopy
import torch
import torch.nn as nn
from torch.autograd import Variable as Var

class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pearson(self, predictions, labels):
        #hack cai nay cho no thanh accuracy
        x = deepcopy(predictions)
        y = deepcopy(labels)
        x -= x.mean()
        x /= x.std()
        y -= y.mean() # FIXME: 'list' object has no attribute 'mean'
                        # label is a list, not tensor
        y /= y.std()
        return torch.mean(torch.mul(x,y))

    def mse(self, predictions, labels):
        x = Var(deepcopy(predictions), volatile=True)
        y = Var(deepcopy(labels), volatile=True)
        return nn.MSELoss()(x,y).data[0]

    def sentiment_accuracy_score(self, predictions, labels, fine_gained = True, num_classes = 3):
        _labels = deepcopy(labels)
        if num_classes == 2:
            _labels[_labels==2] = 1
        correct = (predictions==_labels).sum()
        total = _labels.size(0)
        acc = float(correct)/total
        return acc