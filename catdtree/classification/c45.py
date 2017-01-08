from catdtree import BaseDecisionTree
from scipy import stats

# TODO: stopping criterion - limit information gain; limit depth, other?


class C45(BaseDecisionTree):
    """C4.5 decision tree for classification.

    This class implements a decision tree using the C4.5 algorithm for
    building it.

    References:
        * Quinlan, J. R. C4.5: Programs for Machine Learning. Morgan Kaufmann
          Publishers, 1993.
    """

    def __init__(self, criterion='entropy'):
        """Construct the C4.5 tree.

        Args:
            * criterion: (default='entropy') string. The function to measure
              the quality of a split. (TODO: 'gini')
        """
        BaseDecisionTree.__init__(self)
        self.criterion = criterion

    def _choose_best_split(self, X_part, y_part):
        """Choose the best split according to the C4.5 algorithm.

        Args:
            * X_part: pandas.DataFrame. The data of the independent variables
              which reach the current tree node.
            * y_part: pandas.Series. The data of the dependent variable
              regarding `X_part`.

        Returns:
            A tuple (condition_str, split_filter). For more info see the docs
            of catdtree.TreeNode.__init__.
        """
        def compile_split_filter_and_text(split):
            if split[1]:  # numerical feature
                def split_filter_1(X, y):
                    branch_data_mask = X[split[0]] <= split[1]
                    return (X[branch_data_mask], y[branch_data_mask])

                def split_filter_2(X, y):
                    branch_data_mask = X[split[0]] > split[1]
                    return (X[branch_data_mask], y[branch_data_mask])

                cond_1_str = split[0] + u' <= ' + unicode(split[1])
                cond_2_str = split[0] + u' > ' + unicode(split[1])

                return [(cond_1_str, split_filter_1),
                        (cond_2_str, split_filter_2)]
            else:  # categorical feature
                values = X_part[split[0]].unique()
                compiled_split = []
                for value in values:
                    def split_filter(X, y):
                        branch_data_mask = X[split[0]] == value
                        return (X[branch_data_mask], y[branch_data_mask])
                    cond_str = split[0] + u' is ' + unicode(value)
                    compiled_split.append((cond_str, split_filter))
                return compiled_split

        classes = y_part.unique()
        class_support = [sum(y_part == c) / float(len(y_part)) for c in classes]
        parent_support = len(y_part)
        if self.criterion == 'entropy':
            # compute the entropy of all the data
            parent_entropy = stats.entropy(class_support, base=2)
        else:
            raise NotImplementedError('TODO: gini')

        splits = []

        for feature, dtype in zip(X_part, X_part.dtypes):
            if dtype == object:  # categorical feature
                # Create a branch for each value of the categorical feature.
                values = X_part[feature].unique()
                if len(values) < 2:
                    # we don't want to split on feature containing just one value
                    continue
                branches_entropy = 0  # the accumulated entropy of all branches
                for value in values:
                    branch_data_mask = X_part[feature] == value
                    y_branch = y_part[branch_data_mask]
                    branch_support = len(y_branch)
                    if branch_support == 0:
                        continue
                    branch_class_support = [sum(y_branch == c) / float(branch_support) for c in classes]
                    if self.criterion == 'entropy':
                        branch_entropy = stats.entropy(branch_class_support, base=2)
                        branches_entropy += branch_entropy * float(branch_support) / parent_support
                    else:
                        raise NotImplementedError('TODO: gini')
                info_gain = parent_entropy - branches_entropy
                splits.append((feature, None, info_gain))
            else:  # numerical feature
                # Try out all binary splits of the data over the numerical
                # feature. Choose the best value for the split using
                # information gain.
                split_values = X_part[feature].unique()
                value_splits = []
                for value in split_values:
                    branches_entropy = 0
                    branch_data_mask = X_part[feature] <= value
                    y_child_1 = y_part[branch_data_mask]
                    y_child_2 = y_part[~branch_data_mask]
                    ch_1_support = len(y_child_1)
                    ch_2_support = len(y_child_2)
                    if ch_1_support == 0 or ch_2_support == 0:
                        continue
                    ch_1_class_support = [sum(y_child_1 == c) / float(ch_1_support) for c in classes]
                    ch_2_class_support = [sum(y_child_2 == c) / float(ch_2_support) for c in classes]
                    if self.criterion == 'entropy':
                        child_1_entropy = stats.entropy(ch_1_class_support, base=2)
                        child_2_entropy = stats.entropy(ch_2_class_support, base=2)
                        branches_entropy += child_1_entropy * float(ch_1_support) / parent_support \
                                          + child_2_entropy * float(ch_2_support) / parent_support
                    else:
                        raise NotImplementedError('TODO: gini')
                    info_gain = parent_entropy - branches_entropy
                    value_splits.append((feature, value, info_gain))
                if value_splits:
                    # if there are any valid splits on the numerical feature
                    # get the best value to split on
                    splits.append(max(value_splits, key=lambda x: x[2]))

        if splits:
            # if there is any valid split of the data on the features
            # return the best split
            best_split = max(splits, key=lambda x: x[2])
            return compile_split_filter_and_text(best_split)
