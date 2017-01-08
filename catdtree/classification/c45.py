from catdtree import BaseDecisionTree
from scipy import stats


class C45(BaseDecisionTree):

    def __init__(self, criterion='entropy'):
        BaseDecisionTree.__init__(self)
        self.criterion = criterion

    def _choose_best_split(self, X_part, y_part):
        def compile_split_filter_and_text(split):
            if split[1]:  # numerical
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
            else:  # categorical
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
            parent_entropy = stats.entropy(class_support, base=2)
        else:
            raise NotImplementedError('TODO: gini')

        splits = []

        for feature, dtype in zip(X_part, X_part.dtypes):
            if dtype == object:  # categorical
                values = X_part[feature].unique()
                if len(values) < 2:
                    # we don't want to split on feture containing just one value
                    continue
                branches_entropy = 0
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
            else:  # numerical
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
