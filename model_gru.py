import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import utils

# class GRU(nn.Module):
#     def __init__(self, cuda,in_dim, mem_dim, num_class):
#         super(GRU, self).__init__()
#         self.cudaFlag = cuda
#
#         self.Wz = nn.Linear(in_dim, mem_dim)
#         self.Uz = nn.Linear(mem_dim, mem_dim)
#
#         self.Wr = nn.Linear(in_dim, mem_dim)
#         self.Ur = nn.Linear(mem_dim, mem_dim)
#
#         self.Wh = nn.Linear(in_dim, mem_dim)
#         self.Uh = nn.Linear(mem_dim, mem_dim)
#
#     def forward(self, x, h, r):
#         pass



class TreeSimpleGRU(nn.Module):
    def __init__(self, cuda,in_dim, mem_dim, num_class):
        super(TreeSimpleGRU, self).__init__()
        self.cudaFlag = cuda
        self.gru_cell = nn.GRUCell(in_dim, mem_dim)

    def forward(self, tree, w_emb, tag_emb, rel_emb, training = False):
        loss = Var(torch.zeros(1))  # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        for idx in xrange(tree.num_children):
            _, child_loss = self.forward(tree.children[idx], w_emb, tag_emb, rel_emb, training)
            loss = loss + child_loss
        child_h = self.get_child_states(tree)
        tree.state = self.node_forward(w_emb[tree.idx - 1], child_h)

        if self.output_module != None:
            output = self.output_module.forward(tree.state[0], training)
            tree.output = output
            if training and tree.gold_label != None:
                target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
                if self.cudaFlag:
                    target = target.cuda()
                loss = loss + self.criterion(output, target)

        return tree.state, loss

        pass

    def node_forward(self, word_emb, tag_emb, rel_emb, child_h):
        k = dosomething(child_h)

        compress_x = torch.cat(k, word_emb, tag_emb, rel_emb, dimension=1) # (1, 300*3)


    def get_child_state(self, tree):
        """
        Get child rels, get child k, get child
        :param tree:
        :return:
        """
        if tree.num_children == 0:
            child_h = Var(torch.zeros(1, 1, self.mem_dim))
            if self.cudaFlag:
                child_h = child_h.cuda()
        else:
            child_h = Var(torch.Tensor(tree.num_children, 1, self.mem_dim))
            if self.cudaFlag:
                child_h =  child_h.cuda()
            for idx in xrange(tree.num_children):
                child_h[idx] = tree.children[idx].state
        return child_h

class AT(nn.Module):
    """
    AT(compress_x[v]) := sigmoid(Wa * tanh(Wb * compress_x[v] + bb) + ba)
    """
    def __init__(self, cuda, dim):
        super(AT, self).__init__()
        self.cudaFlag = cuda
        self.dim = dim

        self.Wa = nn.Linear(dim, dim)
        self.Wb = nn.Linear(dim, dim)

        if self.cudaFlag:
            self.Wa = self.Wa.cuda()
            self.Wb = self.Wb.cuda()

    def forward(self, x):
        out = F.sigmoid(self.Wa(F.tanh(self.Wb(x))))
        return out


class GRU_AT(nn.Module):
    def __init__(self, cuda, dim):
        super(GRU_AT, self).__init__()
        self.cudaFlag = cuda
        self.dim = dim

        self.at = AT(cuda, dim)
        self.gru_cell = nn.GRUCell(dim, dim)

    def forward(self, x, h_prev):
        """

        :param x:
        :param h_prev:
        :return: a * m + (1 - a) * h[t-1]
        """
        a = self.at.forward(x)
        m = self.gru_cell(x, h_prev)
        h = a*m + (1 - a) * h_prev
        return h