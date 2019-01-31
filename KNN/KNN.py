import numpy as np
from sklearn import datasets
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X, y = iris.data, iris.target
X = X[:, :2]

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


# plt.title("Iris Data")
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
# plt.show()

class K_NN():
    def __init__(self, k=15):
        self.x_col_max = None
        self.x_col_min = None
        self.X = None
        self.y = None
        self.k = k

    def fit(self, x_train, y_train):
        self.x_col_max = x_train.max(axis=0)
        self.x_col_min = x_train.min(axis=0)

        self.X = (x_train - self.x_col_min) / (self.x_col_max - self.x_col_min)
        self.y = y_train

    def predict(self, x_test):
        if self.X is None or self.y is None:
            raise NameError("模型未训练")

        x_test = (x_test - self.x_col_min) / (self.x_col_max - self.x_col_min)

        num_test = x_test.shape[0]
        num_train = self.X.shape[0]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt(
            np.transpose([np.sum(np.square(x_test), axis=1)]) - 2 * np.dot(x_test, self.X.T) + np.sum(np.square(self.X),
                                                                                                      axis=1))
        y_predict = np.zeros(num_test)
        for i in range(num_test):
            nearest_y = []
            nearest_y = self.y[np.argsort(dists[i])[:self.k]]
            y_predict[i] = np.argmax(np.bincount(nearest_y))

        return y_predict


knn = K_NN(k=7)
knn.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
