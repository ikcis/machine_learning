import numpy as np
import matplotlib.pyplot as plt


class Linear_Regression:
    def __init__(self):
        self.W = None
        self.learning_rate = None
        self.n_iters = None

    def sgd(self, X, Y, m, lam=0):
        # cost = np.dot((np.dot(X, self.W) - Y).T, (np.dot(X, self.W) - Y)) / 2 * m
        cost = np.sum(np.square(np.dot(X, self.W) - Y) + lam * np.sum(np.square(self.W))) / 2 * m
        self.W = self.W - self.learning_rate * np.dot(X.T, (np.dot(X, self.W) - Y)) / m
        return cost

    def fit(self, X, Y, learning_rate=0.01, n_iters=8000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        m = X.shape[0]
        n = X.shape[1]
        self.W = np.random.normal(size=(X.shape[1],))
        cost_list = []
        for i in range(n_iters):
            cost = self.sgd(X, Y, m)
            cost_list.append(cost)
        # plt.plot(np.arange(self.n_iters), cost_list)
        # plt.title("Loss in Training")
        # plt.xlabel("iter_num")
        # plt.ylabel("loss_value")
        # plt.show()

    def score(self, X, Y):
        cost = np.sum(np.square(np.dot(X, self.W) - Y)) / 2 * X.shape[0]
        return cost

    def predict(self, X):
        if self.W is None:
            raise NameError("未经过训练")
        res = np.dot(X, self.W)
        return res

    def get_params(self):
        return self.W


m = 80  # 样本数
n = 7  # 特征数
X_train = np.random.normal(size=(m, n))
W_train_true = np.random.normal(size=(X_train.shape[1],))
Y_no_noise = np.dot(X_train, W_train_true)
Y_train = Y_no_noise + np.random.normal(scale=0.25, size=Y_no_noise.shape)
model = Linear_Regression()
model.fit(X_train, Y_train)

a = model.score(X_train, Y_train)
print(a)
print()

X_test = np.random.normal(size=(10, n))
W_test_true = np.random.normal(size=(X_test.shape[1],))
Y_no_noise_test = np.dot(X_test, W_test_true)
Y_test = Y_no_noise_test + np.random.normal(scale=0.25, size=Y_no_noise_test.shape)

pred_test = model.predict(X_test)

b = model.score(X_test, Y_test)
print(b)
