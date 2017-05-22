import os
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.utils.data as data
from tree import Tree
from vocab import Vocab
import Constants
from itertools import izip

# Dataset class for SICK dataset
class SICKDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes):
        super(SICKDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        self.lsentences = self.read_sentences(os.path.join(path,'a.toks'))
        self.rsentences = self.read_sentences(os.path.join(path,'b.toks'))

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
        label = deepcopy(self.labels[index])
        return (ltree,lsent,rtree,rsent,label)

    def read_sentences(self, filename):
        with open(filename,'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
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
            if i-1 not in trees.keys() and parents[i-1]!=-1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx-1] = tree
                    tree.idx = idx-1
                    #if trees[parent-1] is not None:
                    if parent-1 in trees.keys():
                        trees[parent-1].add_child(tree)
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

# Dataset class for dependency dataset
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
            for tr in new_trees:
                tr.depth()
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


# constituency tree dataset
class SSTConstituencyDataset(data.Dataset):
    def __init__(self, path, vocab, tagvocab, num_classes, fine_grain):
        super(SSTConstituencyDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes
        self.fine_grain = fine_grain
        self.tagvocab = tagvocab


        temp_sentences = self.read_sentences(os.path.join(path,'sents.toks'))
        temp_tags = self.read_tags(os.path.join(path,'ctags.txt'))

        temp_trees = self.read_trees(os.path.join(path,'cparents.txt'), os.path.join(path,'clabels.txt'), temp_tags)

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

            self.trees = new_trees
            self.sentences = new_sentences
            self.tags = new_tags
        else:
            self.trees = temp_trees
            self.sentences = temp_sentences
            self.tags = temp_tags


        for i in xrange(0, len(self.trees)):
            self.labels.append(self.trees[i].gold_label)
        self.labels = torch.Tensor(self.labels) # let labels be tensor

        sorted_lists = sorted(izip(self.trees, self.sentences, self.tags, self.labels), key=lambda x: x[0].depth())
        self.trees, self.sentences, self.tags, self.labels = [[x[i] for x in sorted_lists] for i in range(4)]
        self.size = len(self.trees)



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
        label = deepcopy(self.labels[index])
        rel = None
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

    def read_trees(self, filename_parents, filename_labels, tags):
        pfile = open(filename_parents, 'r') # parent node
        lfile = open(filename_labels, 'r') # label node
        p = pfile.readlines()
        l = lfile.readlines()
        pl = zip(p, l, tags) # (parent, label) tuple
        trees = [self.read_tree(p_line, l_line, tags) for p_line, l_line, tags in tqdm(pl)]

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

    def read_tree(self, line, label_line, tags):
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
                    tree.tag = tags[idx-1]
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

class TreeDataset(data.Dataset):
    def __init__(self):
        self.trees = None
        self.sentences = None
        self.tags = None
        self.labels = None
        self.rels = None
        self.tag_flag = False
        self.rel_flag = False
        self.size = 0

    def __len__(self):
        return self.size

    def sort(self):
        if self.tag_flag and not self.rel_flag:
            sorted_lists = sorted(izip(self.trees, self.sentences, self.tags, self.labels), key=lambda x: x[0].depth())
            self.trees, self.sentences, self.tags, self.labels = [[x[i] for x in sorted_lists] for i in range(4)]
        elif self.tag_flag and self.rel_flag:
            sorted_lists = sorted(izip(self.trees, self.sentences, self.tags, self.rels, self.labels), key=lambda x: x[0].depth())
            self.trees, self.sentences, self.tags, self.rels, self.labels = [[x[i] for x in sorted_lists] for i in range(5)]
        elif not self.tag_flag and self.rel_flag:
            sorted_lists = sorted(izip(self.trees, self.sentences, self.rels, self.labels), key=lambda x: x[0].depth())
            self.trees, self.sentences, self.rels, self.labels = [[x[i] for x in sorted_lists] for i in range(4)]
        elif not self.tag_flag and not self.rel_flag:
            sorted_lists = sorted(izip(self.trees, self.sentences, self.labels), key=lambda x: x[0].depth())
            self.trees, self.sentences, self.labels = [[x[i] for x in sorted_lists] for i in range(3)]

    def __getitem__(self, index):
        # ltree = deepcopy(self.ltrees[index])
        # rtree = deepcopy(self.rtrees[index])
        # lsent = deepcopy(self.lsentences[index])
        # rsent = deepcopy(self.rsentences[index])
        # label = deepcopy(self.labels[index])

        tree = deepcopy(self.trees[index])
        sent = deepcopy(self.sentences[index])
        label = deepcopy(self.labels[index])
        if self.tag_flag:
            tag = deepcopy(self.tags[index])
        else:
            tag = None
        if self.rel_flag:
            rel = deepcopy(self.rels[index])
        else:
            rel = None
        return (tree, sent, tag, rel, label)

def make_subtree(source_dataset, tag_flag, rel_flag):
    """
    Extract all subtree
    :param source_dataset: 
    :return: 
    """
    print ('get subtree')
    tree, sent, tag, rel, label = source_dataset[0]
    trees = []
    sents = []
    labels = []
    rels = None
    tags = None
    if tag_flag:
        tags = []
    if rel_flag:
        rels = []

    def get_subtree(input_tree, trees, sents, tags, rels, labels):
        """
        recursion function to get all subtree and put into list
        :param tree: 
        :return: 
        """
        for child in input_tree.children:
            if child.gold_label != None:
                trees.append(deepcopy(child))
                labels.append(deepcopy(child.gold_label))
                # sent, rel, tag are access by tree.idx
                # so we dublicate whole line
                sents.append(deepcopy(sent))
                if tag_flag:
                    tags.append(deepcopy(tag))
                if rel_flag:
                    rels.append(deepcopy(rel))
            get_subtree(child, trees, sents, tags, rels, labels)

    for i in tqdm(xrange(source_dataset.size)):
        tree, sent, tag, rel, label = source_dataset[i]

        trees.append(deepcopy(tree))
        sents.append(deepcopy(sent))
        labels.append(deepcopy(tree.gold_label))
        if tag_flag:
            tags.append(deepcopy(tag))
        if rel_flag:
            rels.append(deepcopy(rel))
        get_subtree(tree, trees, sents, tags, rels, labels)

        # for child in tree.children:
        #     if child.gold_label != None:
        #         trees.append(deepcopy(child))
        #         labels.append(deepcopy(child.gold_label))
        #         # sent, rel, tag are access by tree.idx
        #         # so we dublicate whole line
        #         sents.append(deepcopy(sent))
        #         if tag_flag:
        #             tags.append(deepcopy(tag))
        #         if rel_flag:
        #             rels.append(deepcopy(rel))

    new_dataset = TreeDataset()
    new_dataset.trees = trees
    new_dataset.sentences = sents
    new_dataset.labels = labels
    new_dataset.tags = tags
    new_dataset.rels = rels
    new_dataset.tag_flag = tag_flag
    new_dataset.rel_flag = rel_flag
    new_dataset.size = len(labels)
    new_dataset.labels = torch.Tensor(labels)
    new_dataset.sort()
    calculate_partition_dataset_by_treedepth(new_dataset)
    return new_dataset

def calculate_partition_dataset_by_treedepth(dataset):
    print ('partition')
    cur_dep = 0
    cur_idx = 0
    part_idx = []
    part_depth = []
    part_idx.append(cur_idx)
    for i in tqdm(xrange(dataset.size)):
        tree = dataset[i][0].depth()
        dep = tree.depth()
        if cur_dep < dep:
            part_depth.append(cur_dep) # dep of last part
            cur_dep = dep # dep of new part
            cur_idx = i # index of new part
            part_idx.append(cur_idx) # start index of new part
    part_depth.append(cur_dep)
    dataset.part_depth = part_depth
    dataset.part_index = part_idx
    return part_depth, part_idx

def partition_dataset(dataset, start, end):
    trees, sents, tags, rels, labels = dataset[start: end]
    new_dataset = TreeDataset()
    new_dataset.trees = trees
    new_dataset.sentences = sents
    new_dataset.labels = labels
    new_dataset.tags = tags
    new_dataset.rels = rels
    new_dataset.tag_flag = dataset.tag_flag
    new_dataset.rel_flag = dataset.rel_flag
    new_dataset.size = len(labels)
    new_dataset.labels = torch.Tensor(labels)
    return new_dataset




