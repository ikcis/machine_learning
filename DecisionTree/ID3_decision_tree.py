import numpy as np


class ID3_Decision_Tree():
    def __init__(self):
        self.ID3_Tree = None

    def fit(self, data_set, feature_name):
        self.ID3_Tree = self.generate_tree(data_set, feature_name)
        return self.ID3_Tree

    def predict(self, test_data, test_feature_name):
        pass

    def cal_shannon_entropy(self, data_set):

    def choose_best_feature(self, data_set):
        feature_len = len(data_set[0])-1
        data_len = len(data_set)
        data_set_entropy =

    def vote(self, categories):
        cate_count = {}
        for cate in categories:
            if cate in cate_count.keys():
                cate_count[cate] += 1
            else:
                cate_count[cate] = 1
        voted_category = max(cate_count.items(), key=lambda x: x[1])[0]
        return voted_category

    def generate_tree(self, data_set, features_name):
        categories = [example[-1] for example in data_set]
        if categories.count(categories[0]) == len(categories):
            return categories[0]
        if len(data_set[0]) == 1:
            return self.vote(categories)

        _, best_feature_id = self.choose_best_feature(data_set)

        best_feature_name = features_name[best_feature_id]


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
