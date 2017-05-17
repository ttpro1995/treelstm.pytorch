import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from torch.autograd import Variable as Var
import utils
from model import SentimentModule

def new_gate(mode, word_dim, tag_dim, mem_dim):
    if mode == 'full':
        We = nn.Linear(word_dim, mem_dim)  # embedding
        Wt = nn.Linear(tag_dim, mem_dim)  # child tag
        Wtp = nn.Linear(tag_dim, mem_dim)  # parent tag
        Uh = nn.Linear(mem_dim, mem_dim)  # h_prev from left
        Uk = nn.Linear(mem_dim, mem_dim)  # k from down
        return We, Wt, Wtp, Uh, Uk
    elif mode == 'node':
        Wt = nn.Linear(tag_dim, mem_dim)  # child tag
        Wtp = nn.Linear(tag_dim, mem_dim)  # parent tag
        Uh = nn.Linear(mem_dim, mem_dim)  # h_prev from left
        Uk = nn.Linear(mem_dim, mem_dim)  # k from down
        return Wt, Wtp, Uh, Uk
    elif mode == 'leaf':
        We = nn.Linear(word_dim, mem_dim)  # embedding
        Wt = nn.Linear(tag_dim, mem_dim)  # child tag
        Wtp = nn.Linear(tag_dim, mem_dim)
        Uh = nn.Linear(mem_dim, mem_dim)  # h_prev from left
        return We, Wt, Wtp, Uh
    elif mode == 'simpleleaf':
        We = nn.Linear(word_dim, mem_dim)  # embedding
        Wt = nn.Linear(tag_dim, mem_dim)  # child tag
        Wtp = nn.Linear(tag_dim, mem_dim)
        return We, Wt, Wtp


class LeafModule(nn.Module):
    """
    Leaf module have e, but do not have q, k
    """
    def __init__(self, cuda, word_dim, tag_dim, mem_dim):
        super(LeafModule, self).__init__()
        self.cudaFlag = cuda
        self.word_dim = word_dim
        self.tag_dim = tag_dim
        self.mem_dim = mem_dim

        self.ie, self.it, self.itp, self.ih = new_gate('leaf', word_dim, tag_dim, mem_dim)
        self.oe, self.ot, self.otp, self.oh = new_gate('leaf', word_dim, tag_dim, mem_dim)
        self.fle, self.flt, self.fltp, self.flh = new_gate('leaf', word_dim, tag_dim, mem_dim)
        self.ue, self.ut, self.utp, self.uh = new_gate('leaf', word_dim, tag_dim, mem_dim)

        if self.cudaFlag:
            self.ie = self.ie.cuda()
            self.it = self.it.cuda()
            self.ih = self.ih.cuda()
            self.itp = self.itp.cuda()

            self.oe = self.oe.cuda()
            self.ot = self.ot.cuda()
            self.oh = self.oh.cuda()
            self.otp = self.otp.cuda()

            self.fle = self.fle.cuda()
            self.flt = self.flt.cuda()
            self.flh = self.flh.cuda()
            self.fltp = self.fltp.cuda()

            self.ue = self.ue.cuda()
            self.ut = self.ut.cuda()
            self.uh = self.uh.cuda()
            self.utp = self.utp.cuda()

    def forward(self, e, tag, tag_parent, h_prev, c_prev):
        i = F.sigmoid(self.ie(e) + self.it(tag) + self.itp(tag_parent) +self.ih(h_prev))
        o = F.sigmoid(self.oe(e) + self.ot(tag) + self.otp(tag_parent) + self.oh(h_prev))
        f_left = F.sigmoid(self.fle(e) + self.flt(tag) + self.fltp(tag_parent) + self.flh(h_prev))
        u = F.tanh(self.ue(e) + self.ut(tag) + self.utp(tag_parent) +  self.uh(h_prev))
        c = i*u + f_left*c_prev
        h = o * F.tanh(c)
        return h, c

class SimpleLeafModule(nn.Module):
    """
    Leaf module have e, but do not have q, k
    """
    def __init__(self, cuda, word_dim, tag_dim, mem_dim):
        super(LeafModule, self).__init__()
        self.cudaFlag = cuda
        self.word_dim = word_dim
        self.tag_dim = tag_dim
        self.mem_dim = mem_dim

        self.ie, self.it, self.itp = new_gate('simpleleaf', word_dim, tag_dim, mem_dim)
        self.oe, self.ot, self.otp = new_gate('simpleleaf', word_dim, tag_dim, mem_dim)
        self.fle, self.flt, self.fltp = new_gate('simpleleaf', word_dim, tag_dim, mem_dim)
        self.ue, self.ut, self.utp = new_gate('simpleleaf', word_dim, tag_dim, mem_dim)

        if self.cudaFlag:
            self.ie = self.ie.cuda()
            self.it = self.it.cuda()
            self.ih = self.ih.cuda()
            self.itp = self.itp.cuda()

            self.oe = self.oe.cuda()
            self.ot = self.ot.cuda()
            self.oh = self.oh.cuda()
            self.otp = self.otp.cuda()

            self.fle = self.fle.cuda()
            self.flt = self.flt.cuda()
            self.flh = self.flh.cuda()
            self.fltp = self.fltp.cuda()

            self.ue = self.ue.cuda()
            self.ut = self.ut.cuda()
            self.uh = self.uh.cuda()
            self.utp = self.utp.cuda()

    def forward(self, e, tag, tag_parent, h_prev, c_prev):
        i = F.sigmoid(self.ie(e) + self.it(tag) + self.itp(tag_parent) +self.ih(h_prev))
        o = F.sigmoid(self.oe(e) + self.ot(tag) + self.otp(tag_parent) + self.oh(h_prev))
        f_left = F.sigmoid(self.fle(e) + self.flt(tag) + self.fltp(tag_parent) + self.flh(h_prev))
        u = F.tanh(self.ue(e) + self.ut(tag) + self.utp(tag_parent) +  self.uh(h_prev))
        c = i*u + f_left*c_prev
        h = o * F.tanh(c)
        return h, c


class NodeModule(nn.Module):
    """
    have q, k but not e
    """
    def __init__(self, cuda, word_dim, tag_dim, mem_dim):
        super(NodeModule, self).__init__()
        self.cudaFlag = cuda
        self.word_dim = word_dim
        self.tag_dim = tag_dim
        self.mem_dim = mem_dim

        self.it, self.itp, self.ih, self.ik = new_gate('node', word_dim, tag_dim, mem_dim)
        self.ot, self.otp, self.oh, self.ok = new_gate('node', word_dim, tag_dim, mem_dim)
        self.flt, self.fltp, self.flh, self.flk = new_gate('node', word_dim, tag_dim, mem_dim)
        self.fdt, self.fdtp, self.fdh, self.fdk = new_gate('node', word_dim, tag_dim, mem_dim)
        self.ut, self.utp, self.uh, self.uk = new_gate('node', word_dim, tag_dim, mem_dim)

        if self.cudaFlag:
            self.it = self.it.cuda()
            self.ih = self.ih.cuda()
            self.itp = self.itp.cuda()
            self.ik = self.ik.cuda()

            self.ot = self.ot.cuda()
            self.oh = self.oh.cuda()
            self.otp = self.otp.cuda()
            self.ok = self.ok.cuda()

            self.flt = self.flt.cuda()
            self.flh = self.flh.cuda()
            self.fltp = self.fltp.cuda()
            self.flk = self.flk.cuda()

            self.fdt = self.fdt.cuda()
            self.fdh = self.fdh.cuda()
            self.fdtp = self.fdtp.cuda()
            self.fdk = self.fdk.cuda()

            self.ut = self.ut.cuda()
            self.uh = self.uh.cuda()
            self.utp = self.utp.cuda()
            self.uk = self.uk.cuda()

    def forward(self, tag, tag_parent, h_prev, c_prev, k, q):
        i = F.sigmoid(self.it(tag) + self.itp(tag_parent) +self.ih(h_prev) + self.ik(k))
        o = F.sigmoid(self.ot(tag) + self.otp(tag_parent) + self.oh(h_prev) + self.ok(k))
        f_left = F.sigmoid(self.flt(tag) + self.fltp(tag_parent) + self.flh(h_prev) + self.flk(k))
        f_down = F.sigmoid(self.fdt(tag) + self.fdtp(tag_parent) + self.fdh(h_prev) + self.fdk(k))
        u = F.tanh(self.ut(tag) + self.utp(tag_parent) +  self.uh(h_prev))
        c = i*u + f_down*q + f_left*c_prev
        h = o * F.tanh(c)
        return h, c


class ConstituencyTreeLSTM(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, mem_dim, criterion):
        super(ConstituencyTreeLSTM, self).__init__()
        self.cudaFlag = cuda
        self.word_dim = word_dim
        self.tag_dim = tag_dim
        self.mem_dim = mem_dim
        self.criterion = criterion

        self.leaf_module = LeafModule(cuda, word_dim, tag_dim, mem_dim)
        self.node_module = NodeModule(cuda, word_dim, tag_dim, mem_dim)
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
        for m in [self.leaf_module, self.node_module]:
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
            # leaf case
            tree.state = self.leaf_module.forward(e)
        else:
            for idx in xrange(tree.num_children):
                _, child_loss = self.forward(tree.children[idx], embs, training)
                loss = loss + child_loss


        if self.output_module != None and tree.num_children != 0:
            output = self.output_module.forward(tree.state[0], training)
            tree.output = output
            if training and tree.gold_label != None:
                target = Var(utils.map_label_to_target_sentiment(tree.gold_label))
                if self.cudaFlag:
                    target = target.cuda()
                loss = loss + self.criterion(output, target)
        return tree.state, loss


    def get_child_state(self, tree):
        lc, lh = tree.children[0].state
        rc, rh = tree.children[1].state
        return lc, lh, rc, rh

###################################################################
class TreeLSTMSentiment(nn.Module):
    def __init__(self, cuda, word_dim, tag_dim, mem_dim, num_classes, criterion):
        super(TreeLSTMSentiment, self).__init__()
        self.cudaFlag = cuda
        self.tree_module = ConstituencyTreeLSTM(cuda, word_dim, tag_dim, mem_dim, criterion)
        self.output_module = SentimentModule(cuda, mem_dim, num_classes, dropout=True)
        self.tree_module.set_output_module(self.output_module)

    def forward(self, tree, embs, tags, training = False):
        tree_state, loss = self.tree_module(tree, embs, tags, training)
        output = tree.output
        return output, loss