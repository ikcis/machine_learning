import numpy as np


class ID3_Decision_Tree():
    def __init__(self):
        self.ID3_Tree = None

    def fit(self, data_set, feature_name):
        pass

    def predict(self, test_data, test_feature_name):
        pass

    def generate_tree(self, data_set, feature_name):


id3_tree_object = ID3_Decision_Tree()

data_set = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]

feature_name = ['color', 'weight']

id3_tree_object.fit(data_set, feature_name)

test_data = [[0, 0],
             [0, 1],
             [1, 1]]

test_feature_name = ['color', 'weight']

id3_tree_object.predict(test_data, test_feature_name)
