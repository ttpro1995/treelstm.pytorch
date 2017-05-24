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
        self.correct_depth = {}
        self.total_depth = {}

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
    def count_depth(self, correct, depth):
        if depth in self.total_depth.keys():
            self.total_depth[depth] +=1
        else:
            self.total_depth[depth] = 1
            self.correct_depth[depth] = 0
        if correct:
            self.correct_depth[depth] += 1

    def getAcc(self):
        acc = {}
        for height in self.total.keys():
            acc[height] = float(self.correct[height]) / self.total[height]
        return acc

    def getAccDepth(self, start, end = -1):
        if end == -1:
            acc = {}
            for depth in self.total_depth.keys():
                acc[depth] = float(self.correct[depth]) / self.total[depth]
            return acc
        else:
            total = 0
            correct = 0
            acc = {}
            for depth in range(start, end+1):
                if depth in self.total_depth.keys():
                    acc[depth] = float(self.correct_depth[depth]) / self.total_depth[depth]
                    total += self.total_depth[depth]
                    correct += self.correct_depth[depth]
                else:
                    acc[depth] = 0
            group_acc = float(correct)/total
            return acc, group_acc

    def printAccDepth(self, start, end = -1):
        acc, group_acc = self.getAccDepth(start, end)
        for key in acc.keys():
            print ('Depth ' + str(key) +' '+ str(self.correct_depth[key]) +'/'+ str(self.total_depth[key]) +' acc ' + str(acc[key]))


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

    metric.count_depth(True, 0)
    metric.count_depth(False, 0)
    metric.count_depth(True, 1)
    metric.count_depth(True, 2)
    metric.count_depth(True, 3)
    metric.count_depth(True, 4)
    metric.count_depth(True, 5)
    metric.count_depth(True, 5)
    metric.count_depth(False, 5)
    acc, group_acc = metric.getAccDepth(0, 5)
    metric.printAccDepth(0,5)