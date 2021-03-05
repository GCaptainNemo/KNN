#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/4 0:18 


from src.topk_problem import MaxHeap
import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self):
        """
        Node类，存储父节点，左节点，右节点，特征及分割点
        """
        self.father = None
        self.left = None
        self.right = None
        self.dimension_choice = None   # 分割维度指标
        self.split = None    # 数据有序对(X, Y)
        self.visited = False

    def __str__(self):
        return "feature: %s, split: %s" % (str(self.dimension_choice), str(self.split))

    @property
    def brother(self):
        """ 获取兄弟节点 """
        if self.father is None:
            ret = None
        else:
            if self.father.left is self:
                ret = self.father.right
            else:
                ret = self.father.left
        return ret


class KDTree:
    def __init__(self):
        self.root = Node()

    def __str__(self):
        ret = []
        i = 0
        que = [(self.root, -1)]
        while que:
            nd, idx_father = que.pop(0)
            ret.append("%d -> %d: %s" % (idx_father, i, str(nd)))
            if nd.left is not None:
                que.append((nd.left, i))
            if nd.right is not None:
                que.append((nd.right, i))
            i += 1
        return "\n".join(ret)

    def build_tree(self, dataX, dataY):
        """
        用dataX中的数据构造KD-tree，循环选择分割的维度。
        :param dataX:
        :param dataY:
        :return:
        """
        dimension_num = max(dataX[0].shape)   # 可能(n, 1)或(1, n)
        nd = self.root
        indexes = range(len(dataX))
        queue = [(nd, indexes)]
        nd.dimension_choice = 0
        while queue:
            nd, index_lst = queue.pop(0)   # 队列，先入先出
            n = len(index_lst)
            if n == 1:
                ### 到达叶子节点
                nd.split = (dataX[index_lst[0]], dataY[index_lst[0]])
                continue
            # dimension_choice = i % dimension_num
            dimension_choice = nd.dimension_choice
            median_idx = self.get_median_index(dataX, index_lst, dimension_choice)
            idxs_left, idxs_right = self.split_index_lst(dataX, index_lst, dimension_choice, median_idx)
            nd.split = (dataX[median_idx], dataY[median_idx])
            # 送入队列
            if idxs_left != []:
                nd.left = Node()
                nd.left.father = nd
                nd.left.dimension_choice = (dimension_choice + 1) % dimension_num
                queue.append((nd.left, idxs_left))
            if idxs_right != []:
                nd.right = Node()
                nd.right.father = nd
                nd.right.dimension_choice = (dimension_choice + 1) % dimension_num
                queue.append((nd.right, idxs_right))

    def split_index_lst(self, X, idxs, dimension_choice, median_idx):
        idxs_split = [[], []]
        split_val = X[median_idx][dimension_choice]
        for idx in idxs:
            if idx == median_idx:
                continue
            xi = X[idx][dimension_choice]
            if xi < split_val:
                idxs_split[0].append(idx)
            else:
                idxs_split[1].append(idx)
        return idxs_split

    def get_median_index(self, dataX, idxs_lst, dimension_choice):
        n = len(idxs_lst)
        k = n // 2
        col = map(lambda i: (i, dataX[i][dimension_choice]), idxs_lst)
        sorted_idxs = map(lambda x: x[0], sorted(col, key=lambda x: x[1]))
        median_idx = list(sorted_idxs)[k]
        return median_idx

    def search(self, Xi, nd):
        """比较目标元素与当前结点的当前feature，访问对应的子节点。直到到达叶子节点，返回该叶子节点。"""
        while nd.left or nd.right:
            if nd.left is None:
                nd = nd.right
            elif nd.right is None:
                nd = nd.left
            else:
                if Xi[nd.dimension_choice] < nd.split[0][nd.dimension_choice]:
                    nd = nd.left
                else:
                    nd = nd.right
        return nd

    def search_1nn(self, search_xi):
        """ 搜索KD Tree中与目标元素距离最近的节点，使用广度优先搜索来实现。 """
        dist_best = float("inf")
        nd_best = self.search(search_xi, self.root)
        que = [(self.root, nd_best)]
        while que:
            node_root, node_cur = que.pop(0)
            while 1:
                dist = np.linalg.norm(search_xi - node_cur.split[0])
                if dist < dist_best:
                    dist_best = dist
                    nd_best = node_cur
                if node_cur is not node_root:
                    nd_bro = node_cur.brother
                    # 找最好节点的兄弟节点
                    if nd_bro is not None:
                        dist_hyper = self.get_hyper_plane_dist(
                            search_xi, node_cur.father)
                        # 如果距离大于超平面距离则说明不可能在当前节点的兄弟节点，剪枝
                        # if dist_hyper < dist:
                        if dist_hyper < dist_best:
                            # 说明以兄弟节点为根节点的子树可能存在更靠近搜索点的点，此时需要加入队列。
                            sub_nd_best = self.search(search_xi, nd_bro)
                            que.append((nd_bro, sub_nd_best))
                    node_cur = node_cur.father
                else:
                    break
        return nd_best

    def linear_search(self, X_lst, search_xi, K, function=None):
        """
        用最大堆解决TopK问题，复杂度O(Nlogk)，验证用kd-tree算法结果可靠性
        :return: 最大的K个数
        """
        function = lambda x: np.linalg.norm(x - search_xi)
        init_heap = X_lst[:K]
        max_heap = MaxHeap(init_heap, function=function)
        max_heap.make_maxheap()
        for index in range(K, len(X_lst)):
            dist = function(X_lst[index])
            if dist < max_heap.function(max_heap.data[0]):
                max_heap.delete()
                max_heap.insert(X_lst[index])
        max_heap.heap_sort()
        return max_heap.data

    def get_hyper_plane_dist(self, Xi, node):
        dc = node.dimension_choice
        return abs(Xi[dc] - node.split[0][dc])

    def get_hyper_plane_vector(self, Xi, node, left=True):
        dc = node.dimension_choice
        if left:
            res = Xi[dc] - node.split[0][dc]
        else:
            res = node.split[0][dc] - Xi[dc]
        return res

    def search_knn(self, search_xi, k):
        """
        kd树中搜索k个最近邻样本，复杂度由于剪枝应该小于O(NlogK)，这里采用深度优秀搜索策略，
        :param search_xi: 搜索样本点
        :param k: 近邻数
        :param dist: 度量方式
        :return: 
        """
        if k <= 0:
            return
        nd = self.root
        self.root.visited = True
        stack = [self.root]
        function = lambda x: np.linalg.norm(x.split[0] - search_xi)
        max_heap = MaxHeap([self.root], function=function)
        max_heap.make_maxheap()
        while nd.left or nd.right:
            if nd.left is None:
                nd = nd.right
            elif nd.right is None:
                nd = nd.left
            else:
                if search_xi[nd.dimension_choice] < nd.split[0][nd.dimension_choice]:
                    nd = nd.left
                else:
                    nd = nd.right
            stack.append(nd)
            nd.visited = True
            dist = function(nd)
            if len(max_heap.data) < k:
                max_heap.insert(nd)
            elif dist < function(max_heap.data[0]):
                max_heap.delete()
                max_heap.insert(nd)
        while stack:
            node_cur = stack.pop()
            left_child = node_cur.left
            if left_child and not left_child.visited:
                left_child.visited = True
                dist = function(left_child)
                if len(max_heap.data) < k:
                    max_heap.insert(left_child)
                    stack.append(left_child)
                elif dist < function(max_heap.data[0]):
                    max_heap.delete()
                    max_heap.insert(left_child)
                    stack.append(left_child)
                else:
                    # dist_hyper = self.get_hyper_plane_dist(
                    #     search_xi, node_cur.parent)
                    dist_hyper = self.get_hyper_plane_vector(
                        search_xi, node_cur, left=True)
                    if dist_hyper < function(max_heap.data[0]):
                        stack.append(left_child)
            right_child = node_cur.right
            if right_child and not right_child.visited:
                right_child.visited = True
                dist = function(right_child)
                if len(max_heap.data) < k:
                    max_heap.insert(right_child)
                    stack.append(right_child)
                elif dist < function(max_heap.data[0]):
                    max_heap.delete()
                    max_heap.insert(right_child)
                    stack.append(right_child)
                else:
                    # dist_hyper = self.get_hyper_plane_dist(
                    #     search_xi, node_cur.parent)
                    dist_hyper = self.get_hyper_plane_vector(
                        search_xi, node_cur, left=True)
                    if dist_hyper < function(max_heap.data[0]):
                        stack.append(right_child)

        return max_heap.heap_sort()


if __name__ == "__main__":
    N = 10
    X = [np.array([np.random.random() * 100 for _ in range(2)]) for _ in range(N)]
    Y = [1 if np.random.random() > 0.5 else 0 for _ in range(N)]
    kd_tree = KDTree()
    kd_tree.build_tree(X, Y)
    print(str(kd_tree))
    # nearest_node = kd_tree.search_1nn(np.array([3, 4, 5]))
    # print("node.split = ", nearest_node.split[0])
    # print(np.linalg.norm(nearest_node.split[0] - np.array([3, 4, 5])))
    K = 5
    Xi = np.array([50, 50])
    max_heap = kd_tree.search_knn(Xi, K)
    data = kd_tree.linear_search(X, Xi, K)
    for i in range(len(max_heap)):
        print("error_array = ", max_heap[i].split[0] - data[i])
        print("error_dist = ", np.linalg.norm(max_heap[i].split[0] - data[i]))

    for i in range(len(X)):
        plt.scatter(X[i][0],
                    X[i][1],
                    c="b")
    # for i in range(len(max_heap)):
    #     plt.scatter(max_heap[i].split[0][0], max_heap[i].split[0][1], c="r")
    for i in range(len(data)):
        plt.scatter(data[i][0], data[i][1], c="r")
    plt.scatter(Xi[0], Xi[1],
                marker="h", s=10, c="y")
    plt.show()



