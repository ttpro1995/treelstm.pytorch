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

    def sentiment_accuracy_score(self, predictions, labels, fine_gained = True):
        correct = (predictions==labels).sum()
        total = labels.size(0)
        acc = float(correct)/total
        return acc

class SubtreeMetric():
    def __init__(self):
        self.correct = {}
        self.total = {}

    def reset(self):
        self.correct = {}
        self.total = {}

    def count(self, correct, height):
        if height in self.total.keys():
            self.total[height] +=1
        else:
            self.total[height] = 1
            self.correct[height] = 0
        if correct:
            self.correct[height] += 1

    def getAcc(self):
        acc = {}
        for height in self.total.keys():
            acc[height] = float(self.correct[height]) / self.total[height]
        return acc

    def printAcc(self):
        acc = self.getAcc()
        for key in acc.keys():
            print ('phrases ' + str(key) +' '+ str(self.correct[key]) +'/'+ str(self.total[key]) +' acc ' + str(acc[key]))

if __name__ == "__main__":
    metric =  SubtreeMetric()
    metric.count(False, 7)
    for i in xrange(15,2, -1):
        metric.count(True, i)
    metric.count(False, 3)
    metric.count(False, 7)

    acc = metric.getAcc()
    metric.printAcc()