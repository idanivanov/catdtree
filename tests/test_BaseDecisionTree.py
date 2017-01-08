
from nose import with_setup
from nose.tools import assert_equal
import pandas as pd
from catdtree import BaseDecisionTree


class MockDecisionTree(BaseDecisionTree):

    def __init__(self):
        BaseDecisionTree.__init__(self)

    def _choose_best_split(self, X_part, y_part):
        if len(set(X_part['Sex'])) > 1:
            branch_data_mask = X_part['Sex'] == 'Female'
            best_split = [
                (
                    u'Sex is Female',
                    lambda X, y: (X[branch_data_mask], y[branch_data_mask])
                ),
                (
                    u'Sex is Male',
                    lambda X, y: (X[~branch_data_mask], y[~branch_data_mask])
                )
            ]
            return best_split
        if any(X_part['Age'] >= 35) and any(X_part['Age'] < 35):
            branch_data_mask = X_part['Age'] >= 35
            best_split = [
                (
                    u'Age is greater than 35',
                    lambda X, y: (X[branch_data_mask], y[branch_data_mask])
                ),
                (
                    u'Age is less than 35',
                    lambda X, y: (X[~branch_data_mask], y[~branch_data_mask])
                )
            ]
            return best_split
        else:
            return None


def test_fit():
    tree_str_exp = u'''Root
|--> Sex is Female
|    |--> Age is greater than 35
|    |--> Age is less than 35
|--> Sex is Male
|    |--> Age is greater than 35
|    |--> Age is less than 35
'''

    hot_data = pd.read_csv('tests/hot.csv')
    X, y = hot_data.drop('Hot', axis=1), hot_data['Hot']
    mdt = MockDecisionTree()
    mdt.fit(X, y)
    tree_str = mdt.tree.show()
    assert tree_str_exp == tree_str, 'The tree was not built as expected.'
