class TreeNode(object):
    def __init__(self, split_filter, condition_str):
        self.split_filter = split_filter
        self.condition_str = condition_str
        self.children = []

    def add_child(self, tree_node):
        self.children.append(tree_node)

    def show(self, level=0):
        assert level >= 0
        if level:
            prefix = u'|    ' * (level - 1) + u'|--> '
        else:
            prefix = u''
        s = prefix + unicode(self) + u'\n'
        for child in self.children:
            s += child.show(level + 1)
        return s

    def __repr__(self):
        return self.condition_str

    def __str__(self):
        return self.condition_str
