#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： 11360
# datetime： 2021/3/4 17:03 

# topk_problem, 即取出N个数据中最大的K个数，如果用排序的话复杂度为O(NlogN)，
# 用最小堆的方式复杂度为O(klogk + NlogK) = O(Nlogk)


# 最小堆解决思路:
# 1. 取列表前k个元素建立一个最小堆，堆顶就是目前第k大的数
# 2. 依次向后遍历原列表，对于列表中的元素，如果小于堆顶，则忽略该元素;
# 3. 如果大于堆顶，则将堆顶更换为该元素，并且对堆进行一次调整；
# 4. 遍历列表所有元素后，进行一个堆排序输出。

import random
import numpy as np


class MinHeap:
    def __init__(self, data, function=lambda x:x):
        """
        最小堆，包括插入、删除、堆排序、构造最小堆功能
        :param data: 处理的数据
        :param function: 用function来提取data中的特征进行堆操作
        """
        self.data = data
        self.function = function

    def make_minheap(self):
        """ 构造一个最小堆，复杂度O(KlogK) """
        max_index = len(self.data) - 1
        for i in range(max_index, -1, -1):
            target_index = i
            while True:
                # 从后往前数据下移
                lst = []
                for j in range(1, 3):
                    if target_index * 2 + j <= max_index:
                        lst.append(self.function(self.data[target_index * 2 + j]))
                if not lst:
                    break
                elif self.function(self.data[target_index]) > np.min(lst):
                    new_index = 2 * target_index + np.argmin(lst) + 1
                    self.data[target_index], self.data[new_index] = \
                        self.data[new_index], self.data[target_index]
                    target_index = new_index
                else:
                    break
        print("mini heap = ", self.data)

    def insert(self, data_piece):
        """ 在最小堆中插入数据，数据上浮，复杂度O(logK) """
        self.data.append(data_piece)
        target_index = len(self.data) - 1
        while True:
            if target_index == 0:
                break
            father_index = (target_index - 1) // 2
            if self.function(self.data[target_index]) < self.function(self.data[father_index]):
                self.data[target_index], self.data[father_index] = self.data[father_index], self.data[target_index]
                target_index = father_index
            else:
                break

    def delete(self):
        """ 在最小堆中删除堆顶元素， 复杂度O(logK)，数据下沉 """
        self.data[0] = self.data[-1]
        self.data.pop()
        target_index = 0
        max_index = len(self.data) - 1
        while True:
            lst = []
            for i in range(1, 3):
                if target_index * 2 + i <= max_index:
                    lst.append(self.function(self.data[target_index * 2 + i]))
            if not lst:
                break
            if self.function(self.data[target_index]) > min(lst):
                new_index = 2 * target_index + np.argmin(lst) + 1
                self.data[target_index], self.data[new_index] = \
                    self.data[new_index], self.data[target_index]
                target_index = new_index
            else:
                break

    def heap_sort(self):
        result = []
        while self.data:
            result.append(self.data[0])
            self.delete()
        return result


class MaxHeap:
    def __init__(self, data, function=lambda x:x):
        """
        最大堆，包括构造最大堆、插入、删除、堆排序功能
        :param data: 处理的数据
        :param function: 用function来提取data的特征构造最大堆
        """
        self.data = data
        self.function = function

    def make_maxheap(self):
        """ 构造一个最大堆，复杂度O(KlogK) """
        max_index = len(self.data) - 1
        for i in range(max_index, -1, -1):
            target_index = i
            while True:
                # 从后往前数据下移
                lst = []
                for j in range(1, 3):
                    if target_index * 2 + j <= max_index:
                        lst.append(self.function(self.data[target_index * 2 + j]))
                if not lst:
                    break
                elif self.function(self.data[target_index]) < np.max(lst):
                    new_index = 2 * target_index + np.argmax(lst) + 1
                    self.data[target_index], self.data[new_index] = \
                        self.data[new_index], self.data[target_index]
                    target_index = new_index
                else:
                    break
        print("maxi heap = ", self.data)

    def insert(self, data_piece):
        """ 在最大堆中插入数据，数据上浮，复杂度O(logK) """
        self.data.append(data_piece)
        target_index = len(self.data) - 1
        while True:
            if target_index == 0:
                break
            father_index = (target_index - 1) // 2
            if self.function(self.data[target_index]) > self.function(self.data[father_index]):
                self.data[target_index], self.data[father_index] = self.data[father_index], self.data[target_index]
                target_index = father_index
            else:
                break

    def delete(self):
        """ 在最大堆中删除堆顶元素， 复杂度O(logK)，数据下沉 """
        self.data[0] = self.data[-1]
        self.data.pop(-1)
        target_index = 0
        max_index = len(self.data) - 1
        while True:
            lst = []
            for i in range(1, 3):
                if target_index * 2 + i <= max_index:
                    lst.append(self.function(self.data[target_index * 2 + i]))
            if not lst:
                break
            if self.function(self.data[target_index]) < np.max(lst):
                new_index = 2 * target_index + np.argmax(lst) + 1
                self.data[target_index], self.data[new_index] = \
                    self.data[new_index], self.data[target_index]
                target_index = new_index
            else:
                break

    def heap_sort(self):
        result = []
        while self.data:
            result.append(self.data[0])
            self.delete()
        return result


class TopK:
    def __init__(self, data, K):
        self.data = data
        self.K = K

    def solve(self, maximize=True):
        """
        :param maximize: True则取前K个最大的；False则取K个最小的
        :return:
        """
        if maximize:
            init_heap = self.data[:self.K].copy()
            function = lambda x:x
            heap = MinHeap(init_heap, function=function)
            heap.make_minheap()
            for i in range(self.K, len(self.data)):
                if function(self.data[i]) > function(heap.data[0]):
                    # 比最小堆最小的元素还大
                    heap.delete()
                    heap.insert(self.data[i])

            print(heap.heap_sort())
        else:
            init_heap = self.data[:self.K].copy()
            heap = MaxHeap(init_heap)
            heap.make_maxheap()
            for i in range(self.K, len(self.data)):
                if self.data[i] < heap.data[0]:
                    # 比最大堆最大的元素还小
                    heap.delete()
                    heap.insert(self.data[i])

            print(heap.heap_sort())

        return heap.data


if __name__ == "__main__":
    data_ = list(range(1000))
    random.shuffle(data_)
    a = TopK(data_, 10)
    a.solve(True)
    # b = minheap(data)
    # b.make_minheap()
    # print(b.heap_sort())

