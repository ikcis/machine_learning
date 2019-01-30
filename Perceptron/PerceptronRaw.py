import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 创建一个数据集，X有两个特征，y={-1，1}
X, y = make_blobs(n_samples=500, centers=2, random_state=6)
y[y == 0] = -1

'''
plt.scatter(X[:, 0], X[:, 1], c=y, s=20)
plt.xlabel("feature_1")
plt.ylabel("feature_2")
plt.show()
'''


class Raw():
    def __init__(self):
        self.W = None
        self.bias = None

    def fit(self, x_train, y_train, learning_rate=0.05, n_iters=100, plot_train=True):
        print("开始训练...")
        num_sample, num_features = x_train.shape
        self.W = np.random.randn(num_features)
        self.bias = 0

        while True:
            erros_example = []
            erros_example_y = []
            # 查找错误分类的样本点
            for idx in range(num_sample):
                example = x_train[idx]
                y_idx = y_train[idx]
                # 计算距离
                distance = y_idx * (np.dot(example, self.W) + self.bias)
                if distance <= 0:
                    erros_example.append(example)
                    erros_example_y.append(y_idx)
            if len(erros_example) == 0:
                break
            else:
                # 随机选择一个错误分类点，修正参数
                random_idx = np.random.randint(0, len(erros_example))
                choosed_example = erros_example[random_idx]
                choosed_example_y = erros_example_y[random_idx]
                self.W = self.W + learning_rate * choosed_example_y * choosed_example
                self.bias = self.bias + learning_rate * choosed_example_y
        print("训练结束")

        # 绘制训练结果部分
        if plot_train is True:
            x_hyperplane = np.linspace(2, 10, 8)
            slope = -self.W[0] / self.W[1]
            intercept = -self.bias / self.W[1]
            y_hpyerplane = slope * x_hyperplane + intercept
            plt.xlabel("feature_1")
            plt.ylabel("feature_2")
            plt.xlim((2, 10))
            plt.ylim((-12, 0))
            plt.title("Dataset and Decision in Training(Raw)")
            plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=20)
            plt.plot(x_hyperplane, y_hpyerplane, color='g', label='Decision_Raw')
            plt.legend(loc='upper left')
            plt.show()

    def predict(self, x):
        if self.W is None or self.bias is None:
            raise NameError("模型未训练")
        y_predict = np.sign(np.dot(x, self.W) + self.bias)
        return y_predict


X_train = X[0:450]
y_train = y[0:450]
X_test = X[450:500]
y_test = y[450:500]

model_raw = Raw()
model_raw.fit(X_train, y_train)

y_predict = model_raw.predict(X_test)
accuracy = np.sum(y_predict == y_test) / y_predict.shape[0]
print("原始形式模型在测试集上的准确率: {0}".format(accuracy))
