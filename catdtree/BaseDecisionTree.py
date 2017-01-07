from sklearn.base import BaseEstimator


# API Standards: http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator


class BaseDecisionTree(BaseEstimator):

    def __init__(self):
        self.tree = {}

    def _choose_best_feature(self):
        raise NotImplementedError('Override this method.')

    def _choose_best_split(self, feature):
        raise NotImplementedError('Override this method.')

    def _add_child_node(self):
        pass

    def _build_tree(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def get_params(self):
        pass

    def set_params(self):
        pass
