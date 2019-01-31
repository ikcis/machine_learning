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



