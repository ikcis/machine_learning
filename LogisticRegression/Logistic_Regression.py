import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(file_name):
    dataMat = []
    labelMat = []
    fr = open(file_name)
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
    return np.array(weights)


# 随机梯度下降
# 梯度下降优化算法在每次更新数据集时都需要遍历整个数据集，计算复杂都较高
# 随机梯度下降一次只用一个样本点来更新回归系数
def stocGradAscent0(dataMatrix, classLabels):
    '''
    Desc:
        随机梯度下降，只使用一个样本点来更新回归系数
    Args:
        dataMatrix -- 输入数据的数据特征（除去最后一列）
        classLabels -- 输入数据的类别标签（最后一列数据）
    Returns:
        weights -- 得到的最佳回归系数
    '''
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    # n*1的矩阵
    # 函数ones创建一个全1的数组
    weights = np.ones(n)  # 初始化长度为n的数组，元素全部为 1
    for i in range(m):
        # sum(dataMatrix[i]*weights)为了求 f(x)的值， f(x)=a1*x1+b2*x2+..+nn*xn,此处求出的 h 是一个具体的数值，而不是一个矩阵
        h = sigmoid(sum(dataMatrix[i] * weights))
        # print 'dataMatrix[i]===', dataMatrix[i]
        # 计算真实类别与预测类别之间的差值，然后按照该差值调整回归系数
        error = classLabels[i] - h
        # 0.01*(1*1)*(1*n)
        # print weights, "*" * 10, dataMatrix[i], "*" * 10, error
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def plotBestFit(dataArr, labelMat, weights):
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


def simpleTest():
    dataMat, labelMat = loadDataSet('./TestSet.txt')
    dataArr = np.array(dataMat)
    # weights = gradAscent(dataArr, labelMat)
    weights = stocGradAscent0(dataArr, labelMat)
    plotBestFit(dataArr, labelMat, weights)


if __name__ == '__main__':
    simpleTest()
