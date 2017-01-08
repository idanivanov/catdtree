class TreeNode(object):
    """A node of the decision tree.

    Each TreeNode object acts as a node in the decision tree. A node is
    characterized by:
        * split_filter: Check __init__.
        * condition_str: Check __init__.
        * children: A list of the children of this node.
    """
    def __init__(self, split_filter, condition_str):
        """Construct the tree node.

        Args:
            * split_filter: function. Put simply, this is the filter which says
              which data will reach this node from its parent and which will
              not. It is a function defined as follows:
                - Input `(X, y)`, where `X` is a pandas.DataFrame of independent
                  variables, and `y` is a pandas.Series of the dependent
                  variable.
                - Output `(X_f, y_f)` are a filtered version of `X` and `y`.
            * condition_str: string. A string representation of the node. This
              string is intended to shortly explain the split_filter in an
              understandable manner.
        """
        self.split_filter = split_filter
        self.condition_str = condition_str
        self.children = []

    def add_child(self, tree_node):
        """Add a child of the current node.

        Args:
            * tree_node: TreeNode. The child.
        """
        self.children.append(tree_node)

    def show(self, level=0):
        """Visualize the subtree rooted at this node recursively.

        Args:
            * level: (default=0) int. The depth of the current recursive call.

        Returns:
            A string visualization of the subtree structure.
        """
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
        """String representation of the node."""
        return self.condition_str

    def __str__(self):
        """String representation of the node."""
        return self.condition_str
