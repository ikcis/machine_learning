import numpy as np
import collections
import re
import time
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

news = fetch_20newsgroups(data_home="./data/", subset='all')


def process_text(data):
    processed_data = []
    for example in data:
        example = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，\n。？、~@#￥%……&*（）]+", " ", example)
        example = re.sub(r"\W", " ", example)
        processed_data.append(example.lower().split())
    return processed_data


def generate_vocab(data_x):
    vocabs = set()
    for example in data_x:
        vocabs = vocabs | set(example)
    return list(vocabs)


def convert_data_to_vec(data_x, vocabs):
    data_vec = np.zeros((len(data_x), len(vocabs)))
    for row, example in enumerate(data_x):
        for col, word in enumerate(example):
            if word in vocabs:
                data_vec[row][col] = 1
    return data_vec


X_train, X_test, Y_train, Y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)

X_train = process_text(X_train)
X_test = process_text(X_test)

vocabs = generate_vocab(X_train[0:2000])
X_train_sub = convert_data_to_vec(X_train[0:2000], vocabs)
X_test_sub = convert_data_to_vec(X_test[0:20], vocabs)

Y_train_sub = Y_train[0:2000]
Y_test_sub = Y_test[0:20]


class NaiveBayesClassifier():
    def __init__(self, lambd=1.0):
        self.lambd = lambd
        self.prior_prob = None
        self.conditional_prob = None

    def fit(self, data_x, data_y):
        start_time = time.time()

        print("开始计算先验概率...")
        cate_num_k = len(set(data_y))
        self.prior_prob = {}
        for cate in set(data_y):
            self.prior_prob[cate] = (data_y.tolist().count(cate) + self.lambd) / (len(data_y) + cate_num_k * self.lambd)

        every_feature_count = []
        for feature_idx in range(data_x.shape[1]):
            feature_value = data_x[:, feature_idx]
            feature_diff_value_count = collections.Counter(feature_value)
            every_feature_count.append(feature_diff_value_count)

        group_data = {}
        for cate in set(data_y):
            sub_data_x = []
            for idx, example_label in enumerate(data_y):
                if example_label == cate:
                    sub_data_x.append(data_x[idx])
            group_data[cate] = np.asarray(sub_data_x)

        print("开始计算条件概率...")

        self.conditional_prob = {}
        for cate in set(data_y):
            cate_data = group_data[cate]
            num_cate = cate_data.shape[0]

            every_feature_cond_prob = []
            for idx in range(cate_data.shape[1]):
                feature_count = every_feature_count[idx]
                cate_feature_value = cate_data[:, idx]

                sj = len(feature_count)

                feature_cond_prob = {}
                for value in feature_count.keys():
                    ajl_count = cate_feature_value.tolist().count(value)
                    ajl_on_cate_prob = (ajl_count + self.lambd) / (num_cate + sj * self.lambd)
                    feature_cond_prob[value] = ajl_on_cate_prob

                every_feature_cond_prob.append(feature_cond_prob)

            self.conditional_prob[cate] = every_feature_cond_prob

        stop_time = time.time()
        print("训练结束，耗时：{0} 秒".format(str(stop_time - start_time)))
        return self.prior_prob, self.conditional_prob

    def predict(self, data_test):

        if self.prior_prob is None or self.conditional_prob is None:
            raise NameError("模型未训练，没有可用的参数")

        test_cate_prob = np.zeros((data_test.shape[0], len(self.prior_prob)))

        cate_idx = ()
        cates_name = []
        for cate in self.prior_prob.keys():
            cate_prior_prob = self.prior_prob[cate]
            every_feature_cond_prob = self.conditional_prob[cate]

            cate_test_data_cond_prob = []

            for example in data_test:
                example_feature_prob = []
                for idx, feature_value in enumerate(example.tolist()):
                    feature_cond_prob = every_feature_cond_prob[idx]
                    if feature_value in feature_cond_prob.keys():
                        example_feature_prob.append(feature_cond_prob[feature_value])
                    else:
                        example_feature_prob.append(1.0)
                cate_test_data_cond_prob.append(example_feature_prob)

            log_cate_union_prob = np.sum(np.log(np.asarray(cate_test_data_cond_prob)), axis=1) + np.log(cate_prior_prob)
            test_cate_prob[:, cate_idx] = log_cate_union_prob
            cates_name.append(cate)
            cate_idx += 1

        argmax_idx = np.argmax(test_cate_prob, axis=1)
        test_cate_result = [cates_name[idx] for idx in argmax_idx]

        return test_cate_result


NBClassifier = NaiveBayesClassifier(1.0)
prior_prob, conditional_prob = NBClassifier.fit(X_train_sub, Y_train_sub)
Y_test_sub_predict = NBClassifier.predict(X_test_sub)
