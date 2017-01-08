from sklearn.base import BaseEstimator
from . import TreeNode


# API Standards: http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator


class BaseDecisionTree(BaseEstimator):
    """Base class representing a decision tree.

    This class inherits the scikit-learn BaseEstimator class. However, it is
    not equivalent to the scikit-learn BaseDecisionTree class. This class is
    supposed to be a completely compatible scikit-learn Estimator.
    """

    def __init__(self):
        """Construct the bese decision tree."""
        self.tree = TreeNode(None, 'Root')

    def _choose_best_split(self, X_part, y_part):
        """Choose the best split for single step of the tree construction.

        This function needs to be overriden by the specific decision tree
        algorithm being used.

        Args:
            * X_part: pandas.DataFrame. The data of the independent variables
              which reach the current tree node.
            * y_part: pandas.Series. The data of the dependent variable
              regarding `X_part`.

        Returns:
            A tuple (condition_str, split_filter). For more info see the docs
            of catdtree.TreeNode.__init__.
        """
        raise NotImplementedError('Override this method.')

    def _split(self, X_part, y_part, tree_node):
        """Recursively construct the decision tree.

        Args:
            * X_part: pandas.DataFrame. The data of the independent variables
              which reach the current `tree_node`.
            * y_part: pandas.Series. The data of the dependent variable
              regarding `X_part`.
            * tree_node: catdtree.TreeNode. The current node where the split is
              considered.
        """
        best_split = self._choose_best_split(X_part, y_part)
        if best_split:
            for condition_str, split_filter in best_split:
                X_part_branch, y_part_branch = split_filter(X_part, y_part)
                tree_node_child = TreeNode(split_filter, condition_str)
                tree_node.add_child(tree_node_child)
                self._split(X_part_branch, y_part_branch, tree_node_child)

    def fit(self, X, y):
        """Construct the decision tree over the given data.

        Args:
            * X: pandas.DataFrame. The data of the independent variables.
            * y: pandas.Series. The data of the dependent variable regarding
              `X`.

        Returns:
            self
        """
        self._split(X, y, self.tree)
        return self

    def predict(self, X):
        """TODO."""
        pass

    def get_params(self):
        """TODO."""
        pass

    def set_params(self):
        """TODO."""
        pass
