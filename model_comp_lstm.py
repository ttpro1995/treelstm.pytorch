import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import utils
import Constants
import const
from model import SentimentModule
from embedding_model import EmbeddingModel


# TODO: Add drop out and attention

class Parent_LSTM(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, mem_dim):
        super(Parent_LSTM, self).__init__()
        self.cudaFlag = cuda
        self.word_dim = word_dim
        self.tag_dim = tag_dim
        self.mem_dim = mem_dim
        self.mem_dim = mem_dim

        self.i_word = nn.Linear(word_dim, mem_dim)
        if self.tag_dim:
            self.i_tag = nn.Linear(tag_dim, mem_dim)
        self.i_h = nn.Linear(mem_dim, mem_dim)

        self.o_word = nn.Linear(word_dim, mem_dim)
        if self.tag_dim:
            self.o_tag = nn.Linear(tag_dim, mem_dim)
        self.o_h = nn.Linear(mem_dim, mem_dim)

        self.f_word = nn.Linear(word_dim, mem_dim)
        if self.tag_dim:
            self.f_tag = nn.Linear(tag_dim, mem_dim)
        self.f_h = nn.Linear(mem_dim, mem_dim)

        self.u_word = nn.Linear(word_dim, mem_dim)
        if self.tag_dim:
            self.u_tag = nn.Linear(tag_dim, mem_dim)
        self.u_h = nn.Linear(mem_dim, mem_dim)

        if self.cudaFlag:
            self.f_word = self.f_word.cuda()
            if self.tag_dim:
                self.f_tag = self.f_tag.cuda()
            self.f_h = self.f_h.cuda()

            self.o_word = self.o_word.cuda()
            if self.tag_dim:
                self.o_tag = self.o_tag.cuda()
            self.o_h = self.o_h.cuda()

            self.u_word = self.u_word.cuda()
            if self.tag_dim:
                self.u_tag = self.u_tag.cuda()
            self.u_h = self.u_h.cuda()

            self.i_word = self.i_word.cuda()
            if self.tag_dim:
                self.i_tag = self.i_tag.cuda()
            self.i_h = self.i_h.cuda()

    def forward(self, word, tag, training=False, h_prev=None, c_prev=None):
        if not h_prev:
            h_prev = Var(torch.zeros(1, self.mem_dim), requires_grad=False)
        if not c_prev:
            c_prev = Var(torch.zeros(1, self.mem_dim), requires_grad=False)

        if self.cudaFlag:
            h_prev = h_prev.cuda()
            c_prev = c_prev.cuda()
        if self.tag_dim:
            i = F.sigmoid(self.i_word(word) + self.i_tag(tag) + self.i_h(h_prev))
            f = F.sigmoid(self.f_word(word) + self.f_tag(tag) + self.f_h(h_prev))
            o = F.sigmoid(self.o_word(word) + self.o_tag(tag) + self.o_h(h_prev))
            u = F.tanh(self.u_word(word) + self.u_tag(tag) + self.u_h(h_prev))
        else:
            i = F.sigmoid(self.i_word(word) + self.i_h(h_prev))
            f = F.sigmoid(self.f_word(word) + self.f_h(h_prev))
            o = F.sigmoid(self.o_word(word) + self.o_h(h_prev))
            u = F.tanh(self.u_word(word) + self.u_h(h_prev))

        c = i * u + f * c_prev
        h = o * F.tanh(c)
        return h, c


class Attention_MLP(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, rel_dim, rel_self, dropout=True):
        super(Attention_MLP, self).__init__()
        self.cudaFlag = cuda
        self.word_dim = word_dim
        self.tag_dim = tag_dim
        self.rel_dim = rel_dim
        self.dropout = dropout
        self.rel_self = rel_self

        self.Wa = nn.Linear(const.attention_hid_dim, 1)
        self.l_word = nn.Linear(word_dim, const.attention_hid_dim)
        if self.tag_dim:
            self.l_tag = nn.Linear(tag_dim, const.attention_hid_dim)
        if self.rel_dim:
            self.l_rel = nn.Linear(rel_dim, const.attention_hid_dim)

        if self.cudaFlag:
            self.Wa = self.Wa.cuda()
            self.l_word = self.l_word.cuda()
            if self.tag_dim:
                self.l_tag = self.l_tag.cuda()
            if self.rel_dim:
                self.l_rel = self.l_rel.cuda()

    def forward(self, word, tag, rel=None, training=False):
        if rel is None and self.rel_dim:
            rel = self.rel_self

        if self.dropout:
            word = F.dropout(word, p=const.p_dropout_input, training=training)
            if self.tag_dim:
                tag = F.dropout(tag, p=const.p_dropout_input, training=training)
            if self.rel_dim:
                rel = F.dropout(rel, p=const.p_dropout_input, training=training)

        if self.tag_dim and self.rel_dim:
            g = F.sigmoid(self.Wa(F.tanh(self.l_word(word) + self.l_tag(tag) + self.l_rel(rel))))

        elif self.tag_dim and not self.rel_dim:
            g = F.sigmoid(self.Wa(F.tanh(self.l_word(word) + self.l_tag(tag))))

        elif not self.tag_dim and self.rel_dim:
            g = F.sigmoid(self.Wa(F.tanh(self.l_word(word) + self.l_rel(rel))))

        elif not self.tag_dim and not self.rel_dim:
            g = F.sigmoid(self.Wa(F.tanh(self.l_word(word))))

        else:
            assert False

        return g

class com_MLP(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, rel_dim, dropout=True):
        super(com_MLP, self).__init__()
        self.cudaFlag = cuda
        self.word_dim = word_dim
        self.tag_dim = tag_dim
        self.rel_dim = rel_dim
        self.dropout = dropout
        self.hid_dim = const.mlp_com_hid_dim
        self.out_dim = const.mlp_com_out_dim
        self.num_layer = const.mlp_num_hid_layer
        self.hid_layers= [] # list of hidden layer (empty if hidden layer = 0)
        self.l_word = nn.Linear(word_dim, const.mlp_com_hid_dim)
        if self.tag_dim:
            self.l_tag = nn.Linear(tag_dim, const.mlp_com_hid_dim)
        if self.rel_dim:
            self.l_rel = nn.Linear(rel_dim, const.mlp_com_hid_dim)

        self.l_last = nn.Linear(const.mlp_com_hid_dim, const.mlp_com_out_dim)

        for i in range(self.num_layer):
            l = nn.Linear(const.mlp_com_hid_dim, const.mlp_com_hid_dim)
            if self.cudaFlag:
                l = l.cuda()
            self.hid_layers.append(l)

        if self.cudaFlag:
            self.l_word = self.l_word.cuda()
            if self.tag_dim:
                self.l_tag = self.l_tag.cuda()
            if self.rel_dim:
                self.l_rel = self.l_rel.cuda()            

    def forward(self, word, tag, rel=None, training=False):
        if self.tag_dim and self.rel_dim:
            h0 = F.tanh(self.l_word(word) + self.l_tag(tag) +self.l_rel(rel))
        elif self.tag_dim and not self.rel_dim:
            h0 = F.tanh(self.l_word(word) + self.l_tag(tag))
        elif not self.tag_dim and self.rel_dim:
            h0 = F.tanh(self.l_word(word) + self.l_rel(rel))
        elif not self.tag_dim and not self.rel_dim:
            h0 = F.tanh(self.l_word(word))

        if self.num_layer == -1:
            return h0
        h = h0
        for l in self.hid_layers:
            h = F.tanh(l(h))
        out = F.tanh(self.l_last(h))
        return out

class CompositionLSTM(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, rel_dim, mem_dim, rel_sel, dropout=True):
        super(CompositionLSTM, self).__init__()
        self.cudaFlag = cuda
        self.word_dim = word_dim
        self.tag_dim = tag_dim
        self.rel_dim = rel_dim
        self.mem_dim = mem_dim
        self.rel_self = rel_sel
        self.dropout = dropout

        # self.fdown_word = nn.Linear(word_dim, mem_dim)
        # if self.tag_dim:
        #     self.fdown_tag = nn.Linear(tag_dim, mem_dim)
        # if self.rel_dim:
        #     self.fdown_rel = nn.Linear(rel_dim, mem_dim)
        self.fdown_mlp = com_MLP(cuda, word_dim, tag_dim, rel_dim)
        self.fdown_k = nn.Linear(mem_dim, mem_dim)
        self.fdown_h = nn.Linear(mem_dim, mem_dim)

        # self.fleft_word = nn.Linear(word_dim, mem_dim)
        # if self.tag_dim:
        #     self.fleft_tag = nn.Linear(tag_dim, mem_dim)
        # if self.rel_dim:
        #     self.fleft_rel = nn.Linear(rel_dim, mem_dim)
        self.fleft_mlp = com_MLP(cuda, word_dim, tag_dim, rel_dim)
        self.fleft_k = nn.Linear(mem_dim, mem_dim)
        self.fleft_h = nn.Linear(mem_dim, mem_dim)

        # self.o_word = nn.Linear(word_dim, mem_dim)
        # if self.tag_dim:
        #     self.o_tag = nn.Linear(tag_dim, mem_dim)
        # if self.rel_dim:
        #     self.o_rel = nn.Linear(rel_dim, mem_dim)
        self.o_mlp = com_MLP(cuda, word_dim, tag_dim, rel_dim)
        self.o_k = nn.Linear(mem_dim, mem_dim)
        self.o_h = nn.Linear(mem_dim, mem_dim)

        # self.u_word = nn.Linear(word_dim, mem_dim)
        # if self.tag_dim:
        #     self.u_tag = nn.Linear(tag_dim, mem_dim)
        # if self.rel_dim:
        #     self.u_rel = nn.Linear(rel_dim, mem_dim)
        self.u_mlp = com_MLP(cuda, word_dim, tag_dim, rel_dim)
        self.u_k = nn.Linear(mem_dim, mem_dim)
        self.u_h = nn.Linear(mem_dim, mem_dim)

        # self.i_word = nn.Linear(word_dim, mem_dim)
        # if self.tag_dim:
        #     self.i_tag = nn.Linear(tag_dim, mem_dim)
        # if self.rel_dim:
        #     self.i_rel = nn.Linear(rel_dim, mem_dim)
        self.i_mlp = com_MLP(cuda, word_dim, tag_dim, rel_dim)
        self.i_k = nn.Linear(mem_dim, mem_dim)
        self.i_h = nn.Linear(mem_dim, mem_dim)

        if self.cudaFlag:
            # self.fdown_word = self.fdown_word.cuda()
            # if self.tag_dim:
            #     self.fdown_tag = self.fdown_tag.cuda()
            # if self.rel_dim:
            #     self.fdown_rel = self.fdown_rel.cuda()
            self.fdown_k = self.fdown_k.cuda()
            self.fdown_h = self.fdown_h.cuda()

            # self.fleft_word = self.fleft_word.cuda()
            # if self.tag_dim:
            #     self.fleft_tag = self.fleft_tag.cuda()
            # if self.rel_dim:
            #     self.fleft_rel = self.fleft_rel.cuda()
            self.fleft_k = self.fleft_k.cuda()
            self.fleft_h = self.fleft_h.cuda()

            # self.o_word = self.o_word.cuda()
            # if self.tag_dim:
            #     self.o_tag = self.o_tag.cuda()
            # if self.rel_dim:
            #     self.o_rel = self.o_rel.cuda()
            self.o_k = self.o_k.cuda()
            self.o_h = self.o_h.cuda()

            # self.u_word = self.u_word.cuda()
            # if self.tag_dim:
            #     self.u_tag = self.u_tag.cuda()
            # if self.rel_dim:
            #     self.u_rel = self.u_rel.cuda()
            self.u_k = self.u_k.cuda()
            self.u_h = self.u_h.cuda()

            # self.i_word = self.i_word.cuda()
            # if self.tag_dim:
            #     self.i_tag = self.i_tag.cuda()
            # if self.rel_dim:
            #     self.i_rel = self.i_rel.cuda()
            self.i_k = self.i_k.cuda()
            self.i_h = self.i_h.cuda()

    # rel_dim > 0 => rel_dim True
    # rel_dim =  => rel_dim False
    def forward(self, word, tag, rel=None, k=None, q=None, h_prev=None, c_prev=None, training=False):
        if rel is None and self.rel_dim:
            # rel = Var(torch.zeros(1, self.rel_dim), requires_grad=False)
            rel = self.rel_self

        if h_prev is None:
            h_prev = Var(torch.zeros(1, self.mem_dim), requires_grad=False)
        if c_prev is None:
            c_prev = Var(torch.zeros(1, self.mem_dim), requires_grad=False)
        if k is None:
            k = Var(torch.zeros(1, self.mem_dim), requires_grad=False)
        if q is None:
            q = Var(torch.zeros(1, self.mem_dim), requires_grad=False)

        if self.cudaFlag:
            h_prev = h_prev.cuda()
            c_prev = c_prev.cuda()
            k = k.cuda()
            q = q.cuda()

        if self.dropout:
            word = F.dropout(word, p=const.p_dropout_input, training=training)
            if self.tag_dim:
                tag = F.dropout(tag, p=const.p_dropout_input, training=training)
            if self.rel_dim:
                rel = F.dropout(rel, p=const.p_dropout_input, training=training)

            k = F.dropout(k, p=const.p_dropout_memory, training=training)
            q = F.dropout(q, p=const.p_dropout_memory, training=training)
            h_prev = F.dropout(h_prev, p=const.p_dropout_memory, training=training)
            c_prev = F.dropout(c_prev, p=const.p_dropout_memory, training=training)

        i = F.sigmoid(self.i_mlp(word, tag, rel, training) \
                      + self.i_h(h_prev) + self.i_k(k))

        f_down = F.sigmoid(self.fdown_mlp(word, tag, rel, training) \
                           + self.fdown_h(h_prev) + self.fdown_k(k))

        f_left = F.sigmoid(self.fleft_mlp(word, tag, rel, training) \
                           + self.fleft_h(h_prev) + self.fleft_k(k))

        o = F.sigmoid(self.o_mlp(word, tag, rel, training) \
                      + self.o_h(h_prev) + self.o_k(k))

        u = F.tanh(self.u_mlp(word, tag, rel) \
                   + self.u_h(h_prev) + self.u_k(k))

        c = i * u + f_down * q + f_left * c_prev

        h = o * F.tanh(c)

        return h, c


class TreeCompositionLSTM(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, criterion,
                 combine_head='mid', rel_self=None, dropout=True, attention = False):
        super(TreeCompositionLSTM, self).__init__()
        self.cudaFlag = cuda
        self.mem_dim = mem_dim
        self.in_dim = word_dim
        self.tag_dim = tag_dim
        self.rel_dim = rel_dim
        self.combine_head = combine_head
        self.dropout = dropout
        self.attention = attention
        if rel_dim and not rel_self:
            rel_self = Var(torch.Tensor(1, self.rel_dim).normal_(-0.05, 0.05))
            if self.cudaFlag:
                rel_self = rel_self.cuda()
        self.rel_self = rel_self

        self.composition_lstm = CompositionLSTM(cuda, word_dim, tag_dim, rel_dim, mem_dim, self.rel_self,
                                                dropout=self.dropout)
        if self.attention:
            self.attention = Attention_MLP(cuda, word_dim, tag_dim, rel_dim, rel_self=self.rel_self, dropout=self.dropout)
        if combine_head != 'mid':
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
            tag = None
            if self.tag_dim:
                tag = tag_emb[tree.idx - 1]

            if self.combine_head == 'start':
                h, c = self.parent_lstm.forward(
                    word_emb[tree.idx - 1], tag, training=training
                )
                for child in tree.children:
                    tag = None
                    rel = None
                    if self.tag_dim:
                        tag = tag_emb[child.idx - 1]
                    if self.rel_dim:
                        rel = rel_emb[child.idx - 1]
                    h, c = self.composition_lstm.forward(
                        word_emb[child.idx - 1], tag, rel, child.state[0],
                        child.state[1], h, c, training=training
                    )
            elif self.combine_head == 'mid':
                list_node = []
                list_node = tree.children
                list_node.append(tree)
                phrase = sorted(list_node, key=lambda k: k.idx)
                h_prev = h
                c_prev = c
                for node in phrase:
                    tag = None
                    rel = None
                    if self.tag_dim:
                        tag = tag_emb[node.idx - 1]
                    if self.rel_dim:
                        rel = rel_emb[node.idx - 1]
                    if node.idx != tree.idx:
                        h, c = self.composition_lstm.forward(
                            word_emb[node.idx - 1], tag, rel, node.state[0],
                            node.state[1], h_prev, c_prev, training=training
                        )
                        if self.attention:
                            g = self.attention.forward(word_emb[node.idx - 1], tag, rel, training=training)
                    else:
                        h, c = self.composition_lstm.forward(
                            word_emb[node.idx - 1], tag, h_prev=h_prev, c_prev=c_prev, training=training
                        )
                        if self.attention:
                            g = self.attention.forward(word_emb[node.idx - 1], tag, training=training)
                    if self.attention:
                        h = torch.mm(g, h) + torch.mm((1 - g), h_prev)
                    h_prev = h
                    c_prev = c

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
    def __init__(self, cuda, in_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, num_classes, criterion, dropout=True):
        super(TreeCompositionLSTMSentiment, self).__init__()
        self.cudaFlag = cuda
        self.tree_module = TreeCompositionLSTM(cuda, in_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, criterion,
                                               dropout=dropout, attention=const.attention)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes, dropout=dropout)
        self.tree_module.set_output_module(self.output_module)

    def get_tree_parameters(self):
        return self.tree_module.getParameters()

    def forward(self, tree, sent_emb, tag_emb, rel_emb, training=False):
        tree_state, loss = self.tree_module(tree, sent_emb, tag_emb, rel_emb, training)
        output = tree.output
        return output, loss
