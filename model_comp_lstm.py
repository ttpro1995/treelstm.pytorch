import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import utils
import Constants
from model import SentimentModule
from embedding_model import EmbeddingModel


class Parent_LSTM(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, mem_dim):
        super(Parent_LSTM, self).__init__()
        self.cudaFlag = cuda

        self.mem_dim = mem_dim

        self.i_word = nn.Linear(word_dim, mem_dim)
        self.i_tag = nn.Linear(tag_dim, mem_dim)
        self.i_h = nn.Linear(mem_dim, mem_dim)

        self.o_word = nn.Linear(word_dim, mem_dim)
        self.o_tag = nn.Linear(tag_dim, mem_dim)
        self.o_h = nn.Linear(mem_dim, mem_dim)

        self.f_word = nn.Linear(word_dim, mem_dim)
        self.f_tag = nn.Linear(tag_dim, mem_dim)
        self.f_h = nn.Linear(mem_dim, mem_dim)

        self.u_word = nn.Linear(word_dim, mem_dim)
        self.u_tag = nn.Linear(tag_dim, mem_dim)
        self.u_h = nn.Linear(mem_dim, mem_dim)

        if self.cudaFlag:
            self.f_word = self.f_word.cuda()
            self.f_tag = self.f_tag.cuda()
            self.f_h = self.f_h.cuda()

            self.o_word = self.o_word.cuda()
            self.o_tag = self.o_tag.cuda()
            self.o_h = self.o_h.cuda()

            self.u_word = self.u_word.cuda()
            self.u_tag = self.u_tag.cuda()
            self.u_h = self.u_h.cuda()

            self.i_word = self.i_word.cuda()
            self.i_tag = self.i_tag.cuda()
            self.i_h = self.i_h.cuda()

    def forward(self, word, tag, training=False, h_prev=None, c_prev=None):
        if not h_prev:
            h_prev = Var(torch.zeros(1, self.mem_dim), requires_grad=False)
        if not c_prev:
            c_prev = Var(torch.zeros(1, self.mem_dim), requires_grad=False)

        if self.cudaFlag:
            h_prev = h_prev.cuda()
            h_prev = h_prev.cuda()

        i = F.sigmoid(self.i_word(word) + self.i_tag(tag) + self.i_h(h_prev))
        f = F.sigmoid(self.f_word(word) + self.f_tag(tag) + self.f_h(h_prev))
        o = F.sigmoid(self.o_word(word) + self.o_tag(tag) + self.o_h(h_prev))

        u = F.tanh(self.u_word(word) + self.u_tag(tag) + self.u_h(h_prev))

        c = i * u + f * c_prev
        h = o * F.tanh(c)
        return h, c


class CompositionLSTM(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, rel_dim, mem_dim):
        super(CompositionLSTM, self).__init__()
        self.cudaFlag = cuda

        self.fdown_word = nn.Linear(word_dim, mem_dim)
        self.fdown_tag = nn.Linear(tag_dim, mem_dim)
        self.fdown_rel = nn.Linear(rel_dim, mem_dim)
        self.fdown_k = nn.Linear(mem_dim, mem_dim)
        self.fdown_h = nn.Linear(mem_dim, mem_dim)

        self.fleft_word = nn.Linear(word_dim, mem_dim)
        self.fleft_tag = nn.Linear(tag_dim, mem_dim)
        self.fleft_rel = nn.Linear(rel_dim, mem_dim)
        self.fleft_k = nn.Linear(mem_dim, mem_dim)
        self.fleft_h = nn.Linear(mem_dim, mem_dim)

        self.o_word = nn.Linear(word_dim, mem_dim)
        self.o_tag = nn.Linear(tag_dim, mem_dim)
        self.o_rel = nn.Linear(rel_dim, mem_dim)
        self.o_k = nn.Linear(mem_dim, mem_dim)
        self.o_h = nn.Linear(mem_dim, mem_dim)

        self.u_word = nn.Linear(word_dim, mem_dim)
        self.u_tag = nn.Linear(tag_dim, mem_dim)
        self.u_rel = nn.Linear(rel_dim, mem_dim)
        self.u_k = nn.Linear(mem_dim, mem_dim)
        self.u_h = nn.Linear(mem_dim, mem_dim)

        self.i_word = nn.Linear(word_dim, mem_dim)
        self.i_tag = nn.Linear(tag_dim, mem_dim)
        self.i_rel = nn.Linear(rel_dim, mem_dim)
        self.i_k = nn.Linear(mem_dim, mem_dim)
        self.i_h = nn.Linear(mem_dim, mem_dim)

        if self.cudaFlag:
            self.fdown_word = self.fdown_word.cuda()
            self.fdown_tag = self.fdown_tag.cuda()
            self.fdown_rel = self.fdown_rel.cuda()
            self.fdown_k = self.fdown_k.cuda()
            self.fdown_h = self.fdown_h.cuda()

            self.fleft_word = self.fleft_word.cuda()
            self.fleft_tag = self.fleft_tag.cuda()
            self.fleft_rel = self.fleft_rel.cuda()
            self.fleft_k = self.fleft_k.cuda()
            self.fleft_h = self.fleft_h.cuda()

            self.o_word = self.o_word.cuda()
            self.o_tag = self.o_tag.cuda()
            self.o_rel = self.o_rel.cuda()
            self.o_k = self.o_k.cuda()
            self.o_h = self.o_h.cuda()

            self.u_word = self.u_word.cuda()
            self.u_tag = self.u_tag.cuda()
            self.u_rel = self.u_rel.cuda()
            self.u_k = self.u_k.cuda()
            self.u_h = self.u_h.cuda()

            self.i_word = self.i_word.cuda()
            self.i_tag = self.i_tag.cuda()
            self.i_rel = self.i_rel.cuda()
            self.i_k = self.i_k.cuda()
            self.i_h = self.i_h.cuda()

    def forward(self, word, tag, rel, k, q, h_prev, c_prev, training=False):
        i = F.sigmoid(self.i_word(word) + self.i_tag(tag) + self.i_rel(rel) \
                      + self.i_h(h_prev) + self.i_k(k))

        f_down = F.sigmoid(self.fdown_word(word) + self.fdown_tag(tag) + self.fdown_rel(rel) \
                           + self.fdown_h(h_prev) + self.fdown_k(k))

        f_left = F.sigmoid(self.fleft_word(word) + self.fleft_tag(tag) + self.fleft_rel(rel) \
                           + self.fleft_h(h_prev) + self.fleft_k(k))

        o = F.sigmoid(self.o_word(word) + self.o_tag(tag) + self.o_rel(rel) \
                      + self.o_h(h_prev) + self.o_k(k))

        u = F.tanh(self.u_word(word) + self.u_tag(tag) + self.u_rel(rel) \
                   + self.u_h(h_prev) + self.u_k(k))

        c = i * u + f_down * q + f_left * c_prev

        h = o * F.tanh(c)

        return h, c


class TreeCompositionLSTM(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, criterion):
        super(TreeCompositionLSTM, self).__init__()
        self.cudaFlag = cuda
        # self.gru_cell = nn.GRUCell(word_dim + tag_dim, mem_dim)
        self.mem_dim = mem_dim
        self.in_dim = word_dim
        self.tag_dim = tag_dim
        self.rel_dim = rel_dim

        self.composition_lstm = CompositionLSTM(cuda, word_dim, tag_dim, rel_dim, mem_dim)
        self.parent_lstm = Parent_LSTM(cuda, word_dim, tag_dim, mem_dim)

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
        for m in []:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return None

    def set_output_module(self, output_module):
        self.output_module = output_module

    def forward(self, tree, w_emb, tag_emb, rel_emb, training=False):
        loss = Var(torch.zeros(1))  # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        for idx in xrange(tree.num_children):
            _, child_loss = self.forward(tree.children[idx], w_emb, tag_emb, rel_emb, training)
            loss = loss + child_loss

        # k, q  = self.get_child_state(tree, w_emb, tag_emb, rel_emb)
        tree.state = self.node_forward(tree, w_emb, tag_emb, rel_emb, training)

        if self.output_module != None:
            output = self.output_module.forward(tree.state[0], training)
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
        c_zero = Var(torch.zeros(1, self.mem_dim), requires_grad=False)
        if self.cudaFlag:
            h_zero = h_zero.cuda()
            c_zero = c_zero.cuda()
        h, c = h_zero, c_zero

        if tree.num_children == 0:
            return h, c
        else:
            h, c = self.parent_lstm.forward(
                word_emb[tree.idx - 1], tag_emb[tree.idx - 1], training=training
            )
            for child in tree.children:
                h, c = self.composition_lstm.forward(
                    word_emb[child.idx - 1], tag_emb[child.idx - 1], rel_emb[child.idx - 1], child.state[0],
                    child.state[1], h, c, training=training
                )

        k = h
        q = c
        return k, q

        # def get_child_state(self, tree, word_emb, tag_emb, rel_emb):
        #     """
        #     Get children word, tag, rel
        #     :param tree: tree we need to get child
        #     :param rels_emb (tensor):
        #     :return:
        #     """
        #     if tree.num_children == 0:
        #         # words = Var(torch.zeros(1, 1, self.in_dim))
        #         # if self.tag_dim:
        #         #     tags = Var(torch.zeros(1, 1, self.tag_dim))
        #         # if self.rel_dim:
        #         #     rels = Var(torch.zeros(1, 1, self.rel_dim))
        #         k = Var(torch.zeros(1, 1, self.mem_dim))
        #         q = Var(torch.zeros(1, 1, self.mem_dim))
        #
        #         if self.cudaFlag:
        #             # words = words.cuda()
        #             # tags = tags.cuda()
        #             # rels = rels.cuda()
        #             k = k.cuda()
        #             q = q.cuda()
        #
        #
        #     else:
        #         # words = Var(torch.Tensor(tree.num_children, 1, self.in_dim))
        #         # rels = Var(torch.Tensor(tree.num_children, 1, self.rel_dim))
        #         # tags = Var(torch.Tensor(tree.num_children, 1, self.tag_dim))
        #
        #
        #
        #         if self.cudaFlag:
        #             # words = words.cuda()
        #             # rels = rels.cuda()
        #             # tags = tags.cuda()
        #             k = k.cuda()
        #             q = q.cuda()
        #
        #
        #         for idx in xrange(tree.num_children):
        #             # words[idx] = word_emb[tree.children[idx].idx - 1]
        #             # rels[idx] = rel_emb[tree.children[idx].idx - 1]
        #             # tags[idx] = tag_emb[tree.children[idx].idx - 1]
        #
        #             k[idx] = tree.children[idx].state[0]
        #             q[idx] = tree.children[idx].state[1]
        #     # return words, tags, rels, k, q
        #     return k, q


class TreeCompositionLSTMSentiment(nn.Module):
    def __init__(self, cuda, in_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, num_classes, criterion):
        super(TreeCompositionLSTMSentiment, self).__init__()
        self.cudaFlag = cuda
        self.tree_module = TreeCompositionLSTM(cuda, in_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, criterion)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes, dropout=True)
        self.tree_module.set_output_module(self.output_module)

    def get_tree_parameters(self):
        return self.tree_module.getParameters()

    def forward(self, tree, sent_emb, tag_emb, rel_emb, training=False):
        tree_state, loss = self.tree_module(tree, sent_emb, tag_emb, rel_emb, training)
        output = tree.output
        return output, loss
