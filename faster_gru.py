import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from torch.autograd import Variable as Var
import utils
from model import SentimentModule

"""
Faster_GRU( k[], p[], tp[]):
    dropout( k[], p[], tp[] )
    h_final = pytorch_GRU( [[k[]; p[]; tp[]] )
    k[tree] = pytorch_GRU_cell( x=0, h_final )

k_leaf = pytorch_GRU_cell( [emb; p[i]]; h=0 )
"""

class FasterGRUTree(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, mem_dim, criterion):
        super(FasterGRUTree, self).__init__()
        self.cudaFlag = cuda
        self.word_dim = word_dim
        self.tag_dim = tag_dim
        self.mem_dim = mem_dim
        self.criterion = criterion

        self.leaf_module = nn.GRUCell(word_dim+tag_dim, mem_dim)
        self.node_module = nn.GRUCell(word_dim+tag_dim, mem_dim)
        self.children_module = nn.GRU(word_dim+tag_dim+tag_dim)
        self.output_module = None

    def set_output_module(self, output_module):
        self.output_module = output_module

    def getParameters(self):
        """
        Get flatParameters
        note that getParameters and parameters is not equal in this case
        getParameters do not get parameters of output module
        :return: 1d tensor
        """
        params = []
        for m in [self.leaf_module, self.node_module, self.children_module]:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params

    def forward(self, tree, embs, tags, training = False):
        # add singleton dimension for future call to node_forward
        # embs = F.torch.unsqueeze(self.emb(inputs),1)

        loss = Var(torch.zeros(1)) # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        if tree.num_children == 0:
            x = torch.cat([embs[tree.idx - 1], tags[tree.idx - 1]])
            h = Var(torch.zeros(1, self.mem_dim), require_grad = False)
            if self.cudaFlag:
                h = h.cuda()
            tree.state = self.leaf_module.forward(x, h)
        else:
            for idx in xrange(tree.num_children):
                _, child_loss = self.forward(tree.children[idx], embs, tags, training)
                loss = loss + child_loss

            x = self.get_child_state(tree, tags)
            _, h_final = self.children_module.forward(x)
            x_zeros = Var(torch.zeros(1, self.word_dim + 2*self.tag_dim), require_grad = False)
            tree.state = self.node_module.forward(x_zeros, h_final)




        if self.output_module != None and tree.num_children != 0:
            output = self.output_module.forward(tree.state[0], training)
            tree.output = output
            if training and tree.gold_label != None:
                target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
                if self.cudaFlag:
                    target = target.cuda()
                loss = loss + self.criterion(output, target)
        return tree.state, loss


    def get_child_state(self, tree, tags):
        """
        get k, tag, parent tag
        :param tree: 
        :return: (n_child, 1, mem_dim + 2*tag_dim)
        """
        n = tree.num_children
        k_tag_tag_parent = Var(torch.zeros(n, 1, self.mem_dim + self.tag_dim*2))
        if self.cudaFlag:
            k_tag_tag_parent = k_tag_tag_parent.cuda()
        for i in xrange(n):
            child_idx = tree.children[i].idx
            k = tree.children[i].state
            t = tags[child_idx-1]
            tp = tags[tree.idx - 1]
            k_t_tp = torch.cat([k, t, tp], 1)
            k_tag_tag_parent[i] = k_t_tp
        return k_tag_tag_parent

###################################################################
class TreeGRUSentiment(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, mem_dim, num_classes, criterion):
        super(TreeGRUSentiment, self).__init__()
        self.cudaFlag = cuda
        self.tree_module = FasterGRUTree(cuda, word_dim, tag_dim, mem_dim, criterion)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes, dropout=True)
        self.tree_module.set_output_module(self.output_module)

    def forward(self, tree, embs, tags, training = False):
        tree_state, loss = self.tree_module(tree, embs, tags, training)
        output = tree.output
        return output, loss
