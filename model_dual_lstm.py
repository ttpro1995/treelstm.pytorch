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

        # sequence lstm for children
        # take h (product by treelstm) concat rel as input =>  child_h_final, treated as child_h_sum in childsumLSTM
        self.seq_lstm = nn.LSTM(input_size=self.mem_dim + self.rel_dim, hidden_size=self.mem_dim, batch_first=False)
        # batch always 1, so let it be (n_child, 1, feature)

        # similar to childsum tree
        self.ix = nn.Linear(self.in_dim,self.mem_dim)
        self.ih = nn.Linear(self.mem_dim,self.mem_dim)

        self.fh = nn.Linear(self.mem_dim, self.mem_dim)
        self.fx = nn.Linear(self.in_dim,self.mem_dim)

        self.ux = nn.Linear(self.in_dim,self.mem_dim)
        self.uh = nn.Linear(self.mem_dim,self.mem_dim)

        self.ox = nn.Linear(self.in_dim,self.mem_dim)
        self.oh = nn.Linear(self.mem_dim,self.mem_dim)


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
        tree.state = self.node_forward(tree, word_tag_emb[tree.idx-1], rel_emb, training)

        if self.output_module != None:
            output = self.output_module.forward(tree.state[0], training)
            tree.output = output
            if training and tree.gold_label != None:
                target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
                if self.cudaFlag:
                    target = target.cuda()
                loss = loss + self.criterion(output, target)
        return tree.state, loss

    def node_forward(self, tree, inputs, rel_emb, training=False):
        """
        words, tags, rels are embedding of child node
        """

        # deal with children first
        if tree.num_children == 0:
            child_h = Var(torch.zeros(1, 1, self.mem_dim))
            child_c = Var(torch.zeros(1, 1, self.mem_dim))
            child_h_final = Var(torch.zeros(1, self.mem_dim))
            # child_h_rel = Var(torch.zeros(1, self.mem_dim + self.rel_dim))
            if self.cudaFlag:
                child_h = child_h.cuda()
                child_c = child_c.cuda()
                child_h_final = child_h_final.cuda()
                # child_h_rel = child_h_rel.cuda()
        else:
            child_h, child_c, child_h_rel  = self.get_child_state(tree, rel_emb)
            output, hn = self.seq_lstm(child_h_rel)
            child_h_final = torch.squeeze(hn[0], 1)


        i = F.sigmoid(self.ix(inputs) + self.ih(child_h_final))
        o = F.sigmoid(self.ox(inputs) + self.oh(child_h_final))
        u = F.tanh(self.ux(inputs) + self.uh(child_h_final))

        # add extra singleton dimension
        fx = F.torch.unsqueeze(self.fx(inputs), 1)
        f = F.torch.cat([self.fh(child_hi) + fx for child_hi in child_h], 0)
        f = F.sigmoid(f)
        # removing extra singleton dimension
        f = F.torch.unsqueeze(f, 1)
        fc = F.torch.squeeze(F.torch.mul(f, child_c), 1)

        c = F.torch.mul(i, u) + F.torch.sum(fc, 0)
        h = F.torch.mul(o, F.tanh(c))


        return h, c

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
                child_h_rel[i] = torch.cat([h, rels], 1)


        # return words, tags, rels, k, q
        return child_h, child_c, child_h_rel


class TreeDualLSTMSentiment(nn.Module):
    def __init__(self, cuda, in_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, num_classes, criterion):
        super(TreeDualLSTMSentiment, self).__init__()
        self.cudaFlag = cuda
        self.tree_module = TreeDualLSTM(cuda, in_dim, tag_dim, rel_dim, mem_dim, at_hid_dim, criterion)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes, dropout=True)
        self.tree_module.set_output_module(self.output_module)

    def get_tree_parameters(self):
        return self.tree_module.getParameters()

    def forward(self, tree, sent_emb, tag_emb, rel_emb, training=False):
        sent_tag = torch.cat([sent_emb, tag_emb], 2)
        tree_state, loss = self.tree_module(tree, sent_tag, rel_emb, training)
        output = tree.output
        return output, loss
