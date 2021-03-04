#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/4 0:18 

# KD-tree 最近邻搜索（广度优先）
# 1. 首先构造一个队列(queue)数据结构
# 2. 从根节点开始，如果输入点在分区面的左边则进入左子节点，在右边则进入右子节点。直到到达叶节点，将该节点当作"当前最佳点"，将(最佳点，
# 根节点)这一有序对数据加入队列。
# 3. 队列出列：
#     a. 如果当前节点比当前最佳点更靠近输入点，则将其变为当前最佳点。
#     b. 检查兄弟节点对应子树能否剪枝，如果不能则将兄弟节点当成根节点，对该子树进行2步骤搜索，把(兄弟节点，叶子节点)有序对加入队列。
#     c. 当前节点 = 当前节点的父节点
#     d. 重复a-c，直到当前节点 = 根节点
# 4. 重复3直到队列为空。



import numpy as np

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

    def build_tree(self, dataX, dataY):
        dimension_num = dataX[0].shape[0]
        i = 0
        nd = self.root
        indexes = range(len(dataX))
        queue = [(nd, indexes)]
        while queue:
            nd, index_lst = queue.pop(0)   # 队列，先入先出
            n = len(index_lst)
            if n == 1:
                ### 到达叶子节点
                nd.split = (dataX[index_lst[0]], dataY[index_lst[0]])
                continue
            dimension_choice = i % dimension_num
            median_idx = self.get_median_index(dataX, index_lst, dimension_choice)
            idxs_left, idxs_right = self.split_index_lst(dataX, index_lst, dimension_choice, median_idx)
            nd.dimension_choice = dimension_choice
            nd.split = (dataX[median_idx], dataY[median_idx])
            # 送入队列
            if idxs_left != []:
                nd.left = Node()
                nd.left.father = nd
                queue.append((nd.left, idxs_left))
            if idxs_right != []:
                nd.right = Node()
                nd.right.father = nd
                queue.append((nd.right, idxs_right))
            i += 1

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
                nd = nd.right
            else:
                if Xi[nd.feature] < nd.split[0][nd.feature]:
                    nd = nd.left
                else:
                    nd = nd.right
        return nd

    def nearest_neighbour_search(self, Xi):
        """ 搜索KD Tree中与目标元素距离最近的节点，使用广度优先搜索来实现。 """
        dist_best = float("inf")
        nd_best = self.search(Xi, self.root)
        que = [(self.root, nd_best)]
        while que:
            nd_root, nd_cur_best = que.pop(0)
            while 1:
                dist = np.linalg.norm(Xi, nd_cur_best)
                if dist < dist_best:
                    dist_best = dist
                    nd_best = nd_cur_best
                if nd_cur_best is not nd_root:
                    nd_bro = nd_cur_best.brother
                    # 找最好节点的兄弟节点
                    if nd_bro is not None:
                        dist_hyper = self.get_hyper_plane_dist(
                            Xi, nd_cur_best.father)
                        # 如果距离大于超平面距离则说明不可能在当前节点的兄弟节点，剪枝
                        if dist_hyper < dist:
                            # 说明以兄弟节点为根节点的子树可能存在更靠近搜索点的点，此时需要加入队列。

                            sub_nd_best = self.search(Xi, nd_bro)
                            que.append((nd_bro, sub_nd_best))
                    nd_cur_best = nd_cur_best.father
                else:
                    break
        return nd_best

    def get_hyper_plane_dist(self, Xi, node):
        dc = node.dimension_choice
        return abs(Xi[dc] - node.split[0][dc])

    def search_knn(self, point, k, dist=None):
        """
        kd树中搜索k个最近邻样本
        :param point: 样本点
        :param k: 近邻数
        :param dist: 度量方式
        :return:
        """

        def search_knn_(kd_node):
            """
            搜索k近邻节点

            :param kd_node: KDNode
            :return: None
            """
            if kd_node is None:
                return
            data = kd_node.data
            distance = p_dist(data)
            if len(heap) < k:
                # 向大根堆中插入新元素
                max_heappush(heap, (kd_node, distance))
            elif distance < heap[0][1]:
                # 替换大根堆堆顶元素
                max_heapreplace(heap, (kd_node, distance))

            axis = kd_node.axis
            if abs(point[axis] - data[axis]) < heap[0][1] or len(heap) < k:
                # 当前最小超球体与分割超平面相交或堆中元素少于k个
                search_knn_(kd_node.left)
                search_knn_(kd_node.right)
            elif point[axis] < data[axis]:
                search_knn_(kd_node.left)
            else:
                search_knn_(kd_node.right)

        if self.root is None:
            raise Exception('kd-tree must be not null.')
        if k < 1:
            raise ValueError("k must be greater than 0.")

        # 默认使用2范数度量距离
        if dist is None:
            p_dist = lambda x: np.linalg.norm(np.array(x) - np.array(point))
        else:
            p_dist = lambda x: dist(x, point)

        heap = []
        search_knn_(self.root)
        return sorted(heap, key=lambda x: x[1])

    def max_heappush(self, heap, new_node, key=lambda x: x[1]):
        """
        大根堆插入元素

        :param heap: 大根堆/列表
        :param new_node: 新节点
        :return: None
        """
        heap.append(new_node)
        pos = len(heap) - 1
        while 0 < pos:
            parent_pos = pos - 1 >> 1
            if key(new_node) <= key(heap[parent_pos]):
                break
            heap[pos] = heap[parent_pos]
            pos = parent_pos
        heap[pos] = new_node

if __name__ == "__main__":
    N = 100000
    X = [[np.random.random() * 100 for _ in range(3)] for _ in range(N)]
    kd_tree = KDTree()
    kd_tree.Xdata = X


