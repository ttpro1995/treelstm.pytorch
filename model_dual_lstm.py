import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
import utils
import Constants
from model import SentimentModule
from embedding_model import EmbeddingModel


class TreeDualLSTM(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, criterion):
        super(TreeDualLSTM, self).__init__()
        self.cudaFlag = cuda
        # self.gru_cell = nn.GRUCell(word_dim + tag_dim, mem_dim)
        self.mem_dim = mem_dim
        self.in_dim = word_dim+tag_dim
        self.tag_dim = tag_dim
        self.rel_dim = rel_dim

        self.seq_lstm = nn.LSTM(input_size=self.mem_dim + self.rel_dim, hidden_size=self.mem_dim, batch_first=False)
        # batch always 1, so let it be (n_child, 1, feature)

        self.ix = nn.Linear(self.in_dim,self.mem_dim)
        self.ih = nn.Linear(self.mem_dim,self.mem_dim)

        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx = nn.Linear(self.in_dim,self.mem_dim)

        self.ux = nn.Linear(self.in_dim,self.mem_dim)
        self.uh = nn.Linear(self.mem_dim,self.mem_dim)

        self.ox = nn.Linear(self.in_dim,self.mem_dim)
        self.oh = nn.Linear(self.mem_dim,self.mem_dim)


        self.criterion = criterion
        if self.cudaFlag:
            self.seq_lstm = self.seq_lstm.cuda()
            self.lstm_cell = self.lstm_cell.cuda()
            self.criterion = self.criterion.cuda()

        self.output_module = None

    def getParameters(self):
        """
        Get flatParameters
        note that getParameters and parameters is not equal in this case
        getParameters do not get parameters of output module
        :return: 1d tensor
        """
        params = []
        for m in [self.seq_lstm, self.lstm_cell]:
            # we do not get param of output module
            l = list(m.parameters())
            params.extend(l)

        one_dim = [p.view(p.numel()) for p in params]
        params = F.torch.cat(one_dim)
        return params

    def set_output_module(self, output_module):
        self.output_module = output_module

    def forward(self, tree, word_tag_emb, rel_emb, training=False):
        loss = Var(torch.zeros(1))  # init zero loss
        if self.cudaFlag:
            loss = loss.cuda()

        for idx in xrange(tree.num_children):
            _, child_loss = self.forward(tree.children[idx], word_tag_emb, rel_emb, training)
            loss = loss + child_loss

        # k, q  = self.get_child_state(tree, w_emb, tag_emb, rel_emb)
        tree.state = self.node_forward(tree, word_tag_emb, rel_emb, training)

        if self.output_module != None:
            output = self.output_module.forward(tree.state[0], training)
            tree.output = output
            if training and tree.gold_label != None:
                target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
                if self.cudaFlag:
                    target = target.cuda()
                loss = loss + self.criterion(output, target)
        return tree.state, loss

    def node_forward(self, tree, word_tag_emb, rel_emb, training=False):
        """
        words, tags, rels are embedding of child node
        """

        # deal with children first
        if tree.children == 0:
            child_h = Var(torch.zeros(1, self.mem_dim))
            child_c = Var(torch.zeros(1, self.mem_dim))
            child_h_rel = Var(torch.zeros(1, self.mem_dim + self.rel_dim))
        else:
            child_h, child_c, child_h_rel  = self.get_child_state(tree, rel_emb)

            output, hn = self.seq_lstm(child_h_rel)


        return k, q

        def get_child_state(self, tree, rel_emb):
            """
            Get children word, tag, rel
            :param tree: tree we need to get child
            :param rel_emb (tensor):
            :return: (1, 1, mem_dim + rel_dim)
            """
            if tree.num_children == 0:
                assert False # never reach here
                # words = Var(torch.zeros(1, 1, self.in_dim))
                # if self.tag_dim:
                # child_h = Var(torch.zeros(1, 1, self.mem_dim))
                # # child_c = Var(torch.zeros(1, 1, self.mem_dim))
                # # if self.rel_dim:
                # rels = Var(torch.zeros(1, 1, self.rel_dim))
                #
                # if self.cudaFlag:
                #     child_h = child_h.cuda()
                #     # child_c = child_c.cuda()
                #     rels = rels.cuda()
                #
                # return child_h, child_c, rels


            else:
                child_h = Var(torch.zeros(tree.num_children, 1, self.mem_dim))
                child_c = Var(torch.zeros(tree.num_children, 1, self.mem_dim))
                child_h_rel = Var(torch.zeros(tree.num_children, 1, self.mem_dim+self.rel_dim)) # concat rel and child h
                # rels = Var(torch.zeros(tree.num_children, 1, self.rel_dim))
                if self.cudaFlag:
                    child_h = child_h.cuda()
                    child_c = child_c.cuda()
                    child_h_rel = child_h_rel.cuda()


                for i in xrange(tree.num_children):
                    # words[idx] = word_emb[tree.children[idx].idx - 1]
                    # rels[idx] = rel_emb[tree.children[idx].idx - 1]
                    # tags[idx] = tag_emb[tree.children[idx].idx - 1]
                    h = tree.children[i].state[0]
                    child_h[i] = h
                    child_c[i] = tree.children[i].state[1]
                    rels = rel_emb[tree.children[i].idx - 1]
                    child_h_rel[i] = torch.cat(h, rels)


            # return words, tags, rels, k, q
            return child_h, child_c, child_h_rel


class TreeCompositionLSTMSentiment(nn.Module):
    def __init__(self, cuda, in_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, num_classes, criterion):
        super(TreeCompositionLSTMSentiment, self).__init__()
        self.cudaFlag = cuda
        self.tree_module = TreeDualLSTM(cuda, in_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, criterion)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes, dropout=True)
        self.tree_module.set_output_module(self.output_module)

    def get_tree_parameters(self):
        return self.tree_module.getParameters()

    def forward(self, tree, sent_emb, tag_emb, rel_emb, training=False):
        tree_state, loss = self.tree_module(tree, sent_emb, tag_emb, rel_emb, training)
        output = tree.output
        return output, loss
