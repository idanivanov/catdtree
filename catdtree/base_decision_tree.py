from sklearn.base import BaseEstimator
from . import TreeNode


# API Standards: http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator


class BaseDecisionTree(BaseEstimator):

    def __init__(self):
        self.tree = TreeNode(None, 'Root')

    def _choose_best_split(self, X_part, y_part):
        raise NotImplementedError('Override this method.')

    def _split(self, X_part, y_part, tree_node):
        best_split = self._choose_best_split(X_part, y_part)
        if best_split:
            for condition_str, split_filter in best_split:
                X_part_branch, y_part_branch = split_filter(X_part, y_part)
                tree_node_child = TreeNode(split_filter, condition_str)
                tree_node.add_child(tree_node_child)
                self._split(X_part_branch, y_part_branch, tree_node_child)

    def fit(self, X, y):
        self._split(X, y, self.tree)

    def predict(self, X):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass
