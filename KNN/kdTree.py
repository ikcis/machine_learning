import numpy as np
from sklearn.neighbors import KDTree


class Node:
    def __init__(self, data, lchild=None, rchild=None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild


class KdTree:
    def __init__(self):
        self.kdtree = None

    def create(self, dataSet, depth):
        if len(dataSet) > 0:
            m, n = np.shape(dataSet)
            mid = int(m / 2)
            axis = depth % n
            sortedDataSet = self.sort(dataSet, axis)
            node = Node(sortedDataSet[mid])
            leftDataSet = sortedDataSet[:mid]
            rightDataSet = sortedDataSet[mid + 1:]
            node.lchild = self.create(leftDataSet, depth + 1)
            node.rchild = self.create(rightDataSet, depth + 1)
            return node
        return None

    def sort(self, dataSet, axis):
        sortDataSet = dataSet[:]
        m, n = np.shape(sortDataSet)
        for i in range(m):
            for j in range(m - i - 1):
                if sortDataSet[j][axis] > sortDataSet[j + 1][axis]:
                    temp = sortDataSet[j]
                    sortDataSet[j] = sortDataSet[j + 1]
                    sortDataSet[j + 1] = temp
        return sortDataSet

    def search(self, tree, x):
        self.nearestPoint = None
        self.nearestValue = 0

        def recursive(node, depth=0):
            if node != None:
                n = len(x)
                axis = depth % n
                if x[axis] < node.data[axis]:
                    recursive(node.lchild, depth + 1)
                else:
                    recursive(node.rchild, depth + 1)

                dist_node_and_x = self.dist(x, node.data)
                if self.nearestPoint is None or self.nearestValue > dist_node_and_x:
                    self.nearestPoint = node.data
                    self.nearestValue = dist_node_and_x

                if abs(x[axis] - node.data[axis]) <= self.nearestValue:
                    if x[axis] < node.data[axis]:
                        recursive(node.rchild, depth + 1)
                    else:
                        recursive(node.lchild, depth + 1)

        recursive(tree)
        return self.nearestPoint

    def dist(self, x1, x2):
        return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5


if __name__ == '__main__':
    dataSet = [[2, 3],
               [5, 4],
               [9, 6],
               [4, 7],
               [8, 1],
               [7, 2]]
    x = [5, 3]
    kdtree = KdTree()
    tree = kdtree.create(dataSet, 0)
    print(kdtree.search(tree, x))
