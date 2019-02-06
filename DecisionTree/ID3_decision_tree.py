import numpy as np


class ID3_Decision_Tree():
    def __init__(self):
        self.ID3_Tree = None

    def fit(self, data_set, feature_name):
        self.ID3_Tree = self.generate_tree(data_set, feature_name)
        return self.ID3_Tree

    def predict(self, test_data, test_feature_name):


    def cal_shannon_entropy(self, data_set):
        label_counts = {}
        data_len = len(data_set)

        for example in data_set:
            label = example[-1]
            if label in label_counts.keys():
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        probs = np.asarray(list(label_counts.values())) / data_len
        shannon_entropy = -np.sum(probs * np.log2(probs))
        return shannon_entropy

    def split_data_set(self, data_set, axis):
        column_single_value = set([example[axis] for example in data_set])
        all_sub_data = []
        for value in column_single_value:
            sub_data = []
            for example in data_set:
                sub_example = []
                if example[axis] == value:
                    sub_example.extend(example[:axis])
                    sub_example.extend(example[axis + 1:])
                    sub_data.append(sub_example)
            all_sub_data.append(sub_data)
        return all_sub_data, column_single_value

    def choose_best_feature(self, data_set):
        feature_len = len(data_set[0]) - 1
        data_len = len(data_set)
        data_set_entropy = self.cal_shannon_entropy(data_set)
        all_feature_info_gain = np.zeros(feature_len)
        for i in range(feature_len):
            all_sub_data, _ = self.split_data_set(data_set, i)
            conditional_entropy = 0
            for sub_data in all_sub_data:
                sub_prob = len(sub_data) / data_len
                sub_enropy = self.cal_shannon_entropy(sub_data)
                conditional_entropy += sub_prob * sub_enropy
            feature_info_gain = data_set_entropy - conditional_entropy
            all_feature_info_gain[i] = feature_info_gain
        best_feature_idx = np.argmax(all_feature_info_gain)
        return all_feature_info_gain, best_feature_idx

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
        ID3_tree = {best_feature_name: {}}
        del features_name[best_feature_id]
        all_sub_data, all_unique_value = self.split_data_set(data_set, best_feature_id)
        for unique_value, sub_data in zip(all_unique_value, all_sub_data):
            sub_features_name = features_name[:]
            ID3_tree[best_feature_name][unique_value] = self.generate_tree(sub_data, sub_features_name)

        return ID3_tree


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
