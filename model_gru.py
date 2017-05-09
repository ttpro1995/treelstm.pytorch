import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import utils
import Constants
from model import SentimentModule
from embedding_model import EmbeddingModel

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
    def __init__(self, cuda, word_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, criterion, leaf_h = None):
        super(TreeSimpleGRU, self).__init__()
        self.cudaFlag = cuda
        self.gru_cell = nn.GRUCell(word_dim + tag_dim, mem_dim)
        self.gru_at = GRU_AT(self.cudaFlag, word_dim + tag_dim + rel_dim + mem_dim, at_hid_dim ,mem_dim)
        self.mem_dim = mem_dim
        self.in_dim = word_dim
        self.tag_dim = tag_dim
        self.rel_dim = rel_dim
        self.leaf_h = leaf_h # init h for leaf node
        if self.leaf_h == None:
            self.leaf_h = Var(torch.rand(1, self.mem_dim))
            torch.save(self.leaf_h, 'leaf_h.pth')

        if self.cudaFlag:
            self.gru_cell = self.gru_cell.cuda()
            self.leaf_h = self.leaf_h.cuda()

        self.criterion = criterion
        self.output_module = None


    def getParameters(self):
        """
        Get flatParameters
        note that getParameters and parameters is not equal in this case
        getParameters do not get parameters of output module
        :return: 1d tensor
        """
        params = []
        for m in [self.gru_cell, self.gru_at]:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params


    def set_output_module(self, output_module):
        self.output_module = output_module

    def forward(self, tree, w_emb, tag_emb, rel_emb, training = False):
        loss = Var(torch.zeros(1))  # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        for idx in xrange(tree.num_children):
            _, child_loss = self.forward(tree.children[idx], w_emb, tag_emb, rel_emb, training)
            loss = loss + child_loss

        if tree.num_children > 0:
            child_rels, child_k  = self.get_child_state(tree, rel_emb)
            tree.state = self.node_forward(w_emb[tree.idx - 1], tag_emb[tree.idx -1], child_rels, child_k)
        elif tree.num_children == 0:
            tree.state = self.leaf_forward(w_emb[tree.idx - 1], tag_emb[tree.idx -1])

        if self.output_module != None:
            output = self.output_module.forward(tree.state, training)
            tree.output = output
            if training and tree.gold_label != None:
                target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
                if self.cudaFlag:
                    target = target.cuda()
                loss = loss + self.criterion(output, target)
        return tree.state, loss


    def leaf_forward(self, word_emb, tag_emb):
        """
        Forward function for leaf node
        :param word_emb:  word embedding of current node u
        :param tag_emb: tag embedding of current node u
        :return: k of current node u
        """
        h = self.leaf_h
        if self.cudaFlag:
            h = h.cuda()
        x = F.torch.cat([word_emb, tag_emb], 1)
        k = self.gru_cell(x, h)
        return k


    def node_forward(self, word_emb, tag_emb, child_rels, child_k):
        """
        Foward function for inner node
        :param word_emb: word embedding of current node u
        :param tag_emb: tag embedding of current node u
        :param child_rels (tensor): rels embedding of child node v
        :param child_k (tensor): k of child node v
        :return:
        """
        n_child = child_k.size(0)
        h = Var(torch.zeros(1, self.mem_dim))
        if self.cudaFlag:
            h = h.cuda()

        for i in range(0, n_child):
            rel = child_rels[i]
            k = child_k[i]
            x = F.torch.cat([word_emb, tag_emb, rel, k], 1)
            h = self.gru_at(x, h)
        k = h

        return k

    def get_child_state(self, tree, rels_emb):
        """
        Get child rels, get child k
        :param tree: tree we need to get child
        :param rels_emb (tensor):
        :return:
        """
        if tree.num_children == 0:
            assert False #  never get here
        else:
            child_k = Var(torch.Tensor(tree.num_children, 1, self.mem_dim))
            child_rels = Var(torch.Tensor(tree.num_children, 1, self.rel_dim))
            if self.cudaFlag:
                child_k = child_k.cuda()
                child_rels = child_rels.cuda()
            for idx in xrange(tree.num_children):
                child_k[idx] = tree.children[idx].state
                child_rels[idx] = rels_emb[tree.children[idx].idx - 1]
        return child_rels, child_k

class AT(nn.Module):
    """
    AT(compress_x[v]) := sigmoid(Wa * tanh(Wb * compress_x[v] + bb) + ba)
    """
    def __init__(self, cuda, in_dim, hid_dim):
        super(AT, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        self.Wa = nn.Linear(hid_dim, 1)
        self.Wb = nn.Linear(in_dim, hid_dim)

        if self.cudaFlag:
            self.Wa = self.Wa.cuda()
            self.Wb = self.Wb.cuda()

    def forward(self, x):
        out = F.sigmoid(self.Wa(F.tanh(self.Wb(x))))
        return out


class GRU_AT(nn.Module):
    def __init__(self, cuda, in_dim, at_hid_dim ,mem_dim):
        super(GRU_AT, self).__init__()
        self.cudaFlag = cuda
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.at = AT(cuda, in_dim, at_hid_dim)
        self.gru_cell = nn.GRUCell(in_dim, mem_dim)

        if self.cudaFlag:
            self.at = self.at.cuda()
            self.gru_cell = self.gru_cell.cuda()

    def forward(self, x, h_prev):
        """

        :param x:
        :param h_prev:
        :return: a * m + (1 - a) * h[t-1]
        """
        a = self.at.forward(x)
        m = self.gru_cell(x, h_prev)
        h = torch.mm(a, m) + torch.mm((1-a), h_prev)
        # h = a*m + (1 - a) * h_prev
        return h

class TreeGRUSentiment(nn.Module):
    def __init__(self, cuda, in_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, num_classes, criterion):
        super(TreeGRUSentiment, self).__init__()
        self.cudaFlag = cuda
        self.tree_module = TreeSimpleGRU(cuda, in_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, criterion)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes, dropout=True)
        self.tree_module.set_output_module(self.output_module)

        # word embeddiing
        # self.word_embedding = nn.Embedding(vocab_size,in_dim,
        #                         padding_idx=Constants.PAD)
        # # embedding for postag and rel
        # self.tag_emb = nn.Embedding(tag_vocabsize, in_dim)
        # self.rel_emb = nn.Embedding(rel_vocabsize, in_dim)

        # self.embedding_model = EmbeddingModel(vocab_size, tag_vocabsize, rel_vocabsize, in_dim)


    def get_tree_parameters(self):
        return self.tree_module.getParameters()

    def forward(self, tree, sent_emb, tag_emb, rel_emb, training = False):
        # sent_emb = F.torch.unsqueeze(self.word_embedding.forward(sent_inputs), 1)
        # tag_emb = F.torch.unsqueeze(self.tag_emb.forward(tag_inputs), 1)
        # rel_emb = F.torch.unsqueeze(self.rel_emb.forward(rel_inputs), 1)
        # sent_emb, tag_emb, rel_emb = self.embedding_model(sent_inputs, tag_inputs, rel_inputs)

        tree_state, loss = self.tree_module(tree, sent_emb, tag_emb, rel_emb, training)
        output = tree.output
        return output, loss

