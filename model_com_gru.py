import config
import torch
from torch import nn
from torch.autograd import Variable as Var
import torch.nn.functional as F
import utils
from model import SentimentModule
import const

class ChildGRU(nn.Module):
    """
    To replace composition lstm
    """
    def __init__(self, cuda, word_dim, tag_dim, rel_dim, mem_dim, rel_sel, dropout=True):
        super(ChildGRU, self).__init__()
        self.args = config.parse_args(type=1)
        self.cudaFlag = cuda
        self.word_dim = word_dim
        self.tag_dim = tag_dim
        self.rel_dim = rel_dim
        self.mem_dim = mem_dim
        self.rel_self = rel_sel
        self.dropout = dropout
        self.gru_cell = nn.GRUCell(word_dim+tag_dim+rel_dim+mem_dim, mem_dim)

        if dropout:
            self.input_dropout = nn.Dropout(p=const.p_dropout_input)
            self.memory_dropout = nn.Dropout(p=const.p_dropout_memory)

    def forward(self, word, tag, rel, k, h_prev, training=False):
        """
        Composition_GRU( [k[i], w[i], p[i], r[i]], h[t-1] ):
        return nn.GRUCell( [k[i], w[i], p[i], r[i]], h[t-1] )
        k[parrent] = h_final
        """

        if self.dropout:
            word = self.input_dropout(word)
            tag = self.input_dropout(tag)
            h_prev = self.memory_dropout(h_prev)
            k = self.memory_dropout(k)

        if self.rel_dim:
            if rel is None:
                rel = Var(self.rel_self)
                if self.cudaFlag:
                    rel = rel.cuda()
            else:
                if self.dropout:
                    rel = self.input_dropout(rel)
            x = torch.cat([word, tag, rel, k], 1)
        else:
            x = torch.cat([word, tag, k], 1)

        h = self.gru_cell.forward(x, h_prev)
        return h


class TreeCompositionGRU(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, criterion,
                 combine_head='mid', rel_self=None, dropout=True, attention = False):
        super(TreeCompositionGRU, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.in_dim = word_dim
        self.tag_dim = tag_dim
        self.rel_dim = rel_dim
        self.combine_head = combine_head
        self.dropout = dropout
        self.attention = attention
        if rel_dim and not rel_self:
            rel_self = torch.Tensor(1, self.rel_dim).normal_(-0.05, 0.05)
        self.rel_self = rel_self

        self.gru = ChildGRU(cuda, word_dim, tag_dim, rel_dim, mem_dim, self.rel_self,
                                                dropout=self.dropout)


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

        l = list(self.parameters())
        params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params

    def getGrad(self):
        params = []

        l = list(self.parameters())
        params.extend(l)

        one_dim = [p.grad.view(p.grad.numel()) for p in params]
        grad = F.torch.cat(one_dim)
        return grad

    def set_output_module(self, output_module):
        self.output_module = output_module

    def forward(self, tree, w_emb, tag_emb, rel_emb, training=False):
        loss = Var(torch.zeros(1))  # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        for idx in xrange(tree.num_children):
            _, child_loss = self.forward(tree.children[idx], w_emb, tag_emb, rel_emb, training)
            loss = loss + child_loss

        tree.state = self.node_forward(tree, w_emb, tag_emb, rel_emb, training)

        if self.output_module != None:
            output = self.output_module.forward(tree.state, training)
            tree.output = output
            if training and tree.gold_label != None:
                target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
                if self.cudaFlag:
                    target = target.cuda()
                loss = loss + self.criterion(output, target)
        return tree.state, loss

    def node_forward(self, tree, word_emb, tag_emb, rel_emb, training=False):
        """
        words, tags, rels are embedding of child node
        """
        h_zero = Var(torch.zeros(1, self.mem_dim), requires_grad=False)
        if self.cudaFlag:
            h_zero = h_zero.cuda()
        h = h_zero

        if tree.num_children == 0:
            return h
        else:

            list_node = tree.children
            list_node.append(tree)
            phrase = sorted(list_node, key=lambda k: k.idx)
            h_prev = h

            for node in phrase:
                tag = None
                rel = None
                if self.tag_dim:
                    tag = tag_emb[node.idx - 1]
                if self.rel_dim:
                    rel = rel_emb[node.idx - 1]
                if node.idx != tree.idx:
                    h = self.gru.forward(
                        word_emb[node.idx - 1], tag, rel, node.state,
                         h_prev, training=training
                    )
                else:
                    # rel => self
                    # no state (no k)
                    rel = Var(self.rel_self)
                    k = Var(torch.zeros(1,self.mem_dim))
                    if self.cudaFlag:
                        rel = rel.cuda()
                        k = k.cuda()
                    h = self.gru.forward(
                        word_emb[node.idx - 1], tag, rel, k, h_prev, training=training
                    )
                h_prev = h

        k = h
        return k
#############################################
class TreeCompositionGRUSentiment(nn.Module):
    def __init__(self, cuda, in_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, num_classes, criterion, dropout=True):
        super(TreeCompositionGRUSentiment, self).__init__()
        self.cudaFlag = cuda
        self.tree_module = TreeCompositionGRU(cuda, in_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, criterion,
                                               dropout=dropout)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes, dropout=dropout)
        self.tree_module.set_output_module(self.output_module)

    def get_tree_parameters(self):
        return self.tree_module.getParameters()

    def forward(self, tree, sent_emb, tag_emb, rel_emb, training=False):
        tree_state, loss = self.tree_module(tree, sent_emb, tag_emb, rel_emb, training)
        output = tree.output
        return output, loss

###################################################
class SimilarityModule(nn.Module):
    def __init__(self, cuda, mem_dim, hidden_dim, num_classes):
        super(SimilarityModule, self).__init__()
        super(SimilarityModule, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)
        self.logsoftmax = nn.LogSoftmax()
        if self.cudaFlag:
            self.wh = self.wh.cuda()
            self.wp = self.wp.cuda()
            self.logsoftmax = self.logsoftmax.cuda()

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)
        out = F.sigmoid(self.wh(vec_dist))
        out = self.logsoftmax(self.wp(out))
        return out

class SimilarityTreeGRU(nn.Module):
    def __init__(self, cuda, vocab_size, word_dim, tag_dim, rel_dim, mem_dim, hidden_dim, num_classes):
        super(SimilarityTreeGRU, self).__init__()
        self.cudaFlag = cuda
        self.tree_module = TreeCompositionGRU(cuda, word_dim, tag_dim, rel_dim, mem_dim, None, criterion=None)
        self.similarity = SimilarityModule(cuda, mem_dim, hidden_dim, num_classes)

    def forward(self, ltree, linputs, ltag, lrel, rtree, rinputs, rtag, rrel):
        lstate, lloss = self.tree_module(ltree, linputs, ltag, lrel)
        rstate, rloss = self.tree_module(rtree, rinputs, rtag, rrel)
        lh = lstate
        rh = rstate
        output = self.similarity(lh, rh)
        return output