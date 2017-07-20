import os
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.utils.data as data
from tree import Tree
from vocab import Vocab
import Constants

class SICKDataset(data.Dataset):
    def __init__(self, path, vocab, tagvocab, relvocab, num_classes):
        super(SICKDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes
        self.tagvocab = tagvocab
        self.relvocab = relvocab

        self.lsentences = self.read_sentences(os.path.join(path,'a.toks'))
        self.rsentences = self.read_sentences(os.path.join(path,'b.toks'))

        self.ltags = self.read_tags(os.path.join(path,'a.tags'))
        self.rtags = self.read_tags(os.path.join(path, 'b.tags'))

        self.lrels = self.read_rels(os.path.join(path, 'a.rels'))
        self.rrels = self.read_rels(os.path.join(path, 'b.rels'))

        self.ltrees = self.read_trees(os.path.join(path,'a.parents'))
        self.rtrees = self.read_trees(os.path.join(path,'b.parents'))

        self.labels = self.read_labels(os.path.join(path,'sim.txt'))

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ltree = deepcopy(self.ltrees[index])
        rtree = deepcopy(self.rtrees[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        ltag = deepcopy(self.ltags[index])
        rtag = deepcopy(self.rtags[index])
        lrel = deepcopy(self.lrels[index])
        rrel = deepcopy(self.rrels[index])
        label = deepcopy(self.labels[index])
        return (ltree, lsent, rtree, rsent, ltag, rtag, lrel, rrel, label)

    def read_sentences(self, filename):
        with open(filename,'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_tags(self, filename):
        with open(filename,'r') as f:
            sentences = [self.read_tag(line) for line in tqdm(f.readlines())]
        return sentences

    def read_rels(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_rel(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_tag(self, line):
        indices = self.tagvocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_rel(self, line):
        indices = self.relvocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_trees(self, filename):
        with open(filename,'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = map(int,line.split())
        trees = dict()
        root = None
        for i in xrange(1,len(parents)+1):
            #if not trees[i-1] and parents[i-1]!=-1:
            if i not in trees.keys() and parents[i-1]!=-1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx] = tree
                    tree.idx = idx
                    #if trees[parent-1] is not None:
                    if parent in trees.keys():
                        trees[parent].add_child(tree)
                        break
                    elif parent==0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        with open(filename,'r') as f:
            labels = map(lambda x: float(x), f.readlines())
            labels = torch.Tensor(labels)
        return labels


# Dataset class for SICK dataset
class SSTDataset(data.Dataset):
    def __init__(self, path, vocab, tagvocab, relvocab , num_classes, fine_grain):
        super(SSTDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes
        self.fine_grain = fine_grain
        self.tagvocab = tagvocab
        self.relvocab = relvocab

        temp_sentences = self.read_sentences(os.path.join(path,'sents.toks'))
        temp_tags = self.read_tags(os.path.join(path,'tags.txt'))
        temp_rels = self.read_rels(os.path.join(path, 'rels.txt'))

        temp_trees = self.read_trees(os.path.join(path,'dparents.txt'), os.path.join(path,'dlabels.txt'), temp_tags, temp_rels)

        # self.labels = self.read_labels(os.path.join(path,'dlabels.txt'))
        self.labels = []

        if not self.fine_grain:
            # only get pos or neg
            new_trees = []
            new_sentences = []
            new_tags = []
            new_rels = []
            for i in range(len(temp_trees)):
                if temp_trees[i].gold_label != 1: # 0 neg, 1 neutral, 2 pos
                    new_trees.append(temp_trees[i])
                    new_sentences.append(temp_sentences[i])
                    new_tags.append(temp_tags[i])
                    new_rels.append(temp_rels[i])
            self.trees = new_trees
            self.sentences = new_sentences
            self.tags = new_tags
            self.rels = new_rels
        else:
            self.trees = temp_trees
            self.sentences = temp_sentences

        for i in xrange(0, len(self.trees)):
            self.labels.append(self.trees[i].gold_label)
        self.labels = torch.Tensor(self.labels) # let labels be tensor
        self.size = len(self.trees)
        for tree in self.trees:
            tree.get_max_n_child()
            tree.depth()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        # ltree = deepcopy(self.ltrees[index])
        # rtree = deepcopy(self.rtrees[index])
        # lsent = deepcopy(self.lsentences[index])
        # rsent = deepcopy(self.rsentences[index])
        # label = deepcopy(self.labels[index])
        tree = deepcopy(self.trees[index])
        sent = deepcopy(self.sentences[index])
        tag = deepcopy(self.tags[index])
        rel = deepcopy(self.rels[index])
        label = deepcopy(self.labels[index])
        return (tree, sent, tag, rel, label)

    def read_sentences(self, filename):
        with open(filename,'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_tags(self, filename):
        with open(filename,'r') as f:
            sentences = [self.read_tag(line) for line in tqdm(f.readlines())]
        return sentences

    def read_rels(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_rel(line) for line in tqdm(f.readlines())]
        return sentences


    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_tag(self, line):
        indices = self.tagvocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_rel(self, line):
        indices = self.relvocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_trees(self, filename_parents, filename_labels, tags, rels):
        pfile = open(filename_parents, 'r') # parent node
        lfile = open(filename_labels, 'r') # label node
        p = pfile.readlines()
        l = lfile.readlines()
        pl = zip(p, l, tags, rels) # (parent, label) tuple
        trees = [self.read_tree(p_line, l_line, tags, rels) for p_line, l_line, tags, rels in tqdm(pl)]

        return trees

    def parse_dlabel_token(self, x):
        if x == '#':
            return None
        else:
            if self.fine_grain: # -2 -1 0 1 2 => 0 1 2 3 4
                return int(x)+2
            else: # # -2 -1 0 1 2 => 0 1 2
                tmp = int(x)
                if tmp < 0:
                    return 0
                elif tmp == 0:
                    return 1
                elif tmp >0 :
                    return 2

    def read_tree(self, line, label_line, tags, rels):
        # trees is dict. So let it be array base 1
        parents = map(int,line.split()) # split each number and turn to int
        trees = dict()
        root = None
        labels = map(self.parse_dlabel_token, label_line.split())
        for i in xrange(1,len(parents)+1):
            #if not trees[i-1] and parents[i-1]!=-1:
            if i not in trees.keys() and parents[i-1]!=-1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx] = tree
                    tree.idx = idx
                    tree.gold_label = labels[idx-1] # add node label
                    tree.tags = tags[idx-1]
                    tree.rels = rels[idx-1]
                    #if trees[parent-1] is not None:
                    if parent in trees.keys():
                        trees[parent].add_child(tree)
                        break
                    elif parent==0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_labels(self, filename):
        # Not in used
        with open(filename,'r') as f:
            labels = map(lambda x: float(x), f.readlines())
            labels = torch.Tensor(labels)
        return labels