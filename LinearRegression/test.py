import numpy as np
import matplotlib.pyplot as plt

m = 80  # 样本数
n = 7  # 特征数
X_train = np.random.normal(size=(m, n))
W_train_true = np.random.normal(size=(X_train.shape[1], 1))
Y_no_noise = np.dot(X_train, W_train_true)
Y_train = Y_no_noise + np.random.normal(scale=0.25, size=Y_no_noise.shape)
print(type(X_train))
X_train.T * (np.dot(X_train, W_train_true) - Y_train)
