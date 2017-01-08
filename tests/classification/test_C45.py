from nose import with_setup
from nose.tools import assert_equal
import pandas as pd
from catdtree.classification import C45


def test_fit():
    tree_str_exp = u'''Root
|--> Weight <= 58
|    |--> Eye Color is Green
|    |--> Eye Color is Blue
|--> Weight > 58
|    |--> Height <= 1.9
|    |    |--> Money in Bank <= 32456
|    |    |    |--> Age <= 32
|    |    |    |    |--> Sex is Female
|    |    |    |    |    |--> Age <= 28
|    |    |    |    |    |    |--> Eye Color is Brown
|    |    |    |    |    |    |--> Eye Color is Blue
|    |    |    |    |    |    |--> Eye Color is Green
|    |    |    |    |    |--> Age > 28
|    |    |    |    |--> Sex is Male
|    |    |    |    |    |--> Age <= 28
|    |    |    |    |    |    |--> Eye Color is Brown
|    |    |    |    |    |    |--> Eye Color is Blue
|    |    |    |    |    |    |--> Eye Color is Green
|    |    |    |    |    |--> Age > 28
|    |    |    |--> Age > 32
|    |    |    |    |--> Sex is Female
|    |    |    |    |    |--> Eye Color is Blue
|    |    |    |    |    |--> Eye Color is Green
|    |    |    |    |--> Sex is Male
|    |    |    |    |    |--> Eye Color is Blue
|    |    |    |    |    |--> Eye Color is Green
|    |    |--> Money in Bank > 32456
|    |    |    |--> Eye Color is Brown
|    |    |    |--> Eye Color is Blue
|    |--> Height > 1.9
|    |    |--> Eye Color is Brown
|    |    |--> Eye Color is Blue
'''
    hot_data = pd.read_csv('tests/hot.csv')
    X, y = hot_data.drop('Hot', axis=1), hot_data['Hot']
    model = C45()
    model.fit(X, y)
    tree_str = model.tree.show()
    assert tree_str_exp == tree_str, 'The tree was not built as expected.'
