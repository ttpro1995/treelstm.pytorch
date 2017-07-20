# tree object from stanfordnlp/treelstm
class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()
        self.gold_label = None # node label for SST
        self.output = None # output node for SST
        self.max_n_child = None # maximum n child of a node in tree
        self._depth = None

    def get_max_n_child(self):
        if self.max_n_child is not None:
            return self.max_n_child

        n_child_list = []
        n_child_list.append(self.num_children)
        for child in self.children:
            n_child_list.append(child.get_max_n_child())
        max_child = max(n_child_list)
        return max_child


    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in xrange(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if self._depth != None:
            return self._depth
        count = 0
        if self.num_children>0:
            for i in xrange(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

