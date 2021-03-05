#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/4 0:17 

from src.kd_tree import KDTree

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from collections import Counter

class MyKNNclassifier:
    def __init__(self, n_neighbour):
        self.kd_tree = KDTree()
        self.K_n = n_neighbour

    def fit(self, X_train, Y_train):
        """
        用训练数据构建KD-tree
        :param X_train: 训练数据X
        :param Y_train: 训练数据类别(label)
        :return: None
        """
        self.kd_tree.build_tree(X_train, Y_train)

    def predict(self, X_test):
        res = []
        for x in X_test:
            node_lst = self.kd_tree.search_knn(x, self.K_n)
            vote_counter = Counter([node_lst[i].split[1] for i in range(self.K_n)])
            class_type = vote_counter.most_common(1)[0][0]
            res.append(class_type)
        return res


if __name__ == "__main__":
    # 设置随机种子，不设置的话默认是按系统时间作为参数，因此每次调用随机模块时产生的随机数都不一样。
    # 设置后每次产生的一样
    np.random.seed(0)
    iris = datasets.load_iris()
    iris_x = iris.data
    iris_y = iris.target
    indices = np.random.permutation(len(iris_x))
    iris_x_train = iris_x[indices[:-10]]
    iris_y_train = iris_y[indices[:-10]]
    iris_x_test = iris_x[indices[-10:]]
    iris_y_test = iris_y[indices[-10:]]
    K = 3
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(iris_x_train, iris_y_train)
    iris_y_predict = knn.predict(iris_x_test)
    print(iris_y_predict)

    obj = MyKNNclassifier(K)
    obj.fit(iris_x_train, iris_y_train)
    res = obj.predict(iris_x_test)
    print(res)