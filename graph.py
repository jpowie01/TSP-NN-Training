import random
import math
import itertools
import sys


class Graph:
    def __init__(self, n):
        self.shortest_tsp = sys.float_info.max
        self.order_tsp = []
        self.order_tsp_list = []
        self.n = n
        self.graph = []
        for i in range(n):
            x = random.random()
            y = random.random()
            self.graph.append((x, y))
        self.each_to_each = [[0 for x in range(self.n)] for y in range(self.n)]
        self.compute_each_to_each()

    def compute_each_to_each(self):
        for i in range(self.n):
            for j in range(i, self.n):
                if i == j:
                    self.each_to_each[i][j] = 0
                else:
                    diff_x = self.graph[i][0] - self.graph[j][0]
                    diff_y = self.graph[i][1] - self.graph[j][1]
                    self.each_to_each[i][j] = self.each_to_each[j][i] = math.sqrt(diff_x * diff_x + diff_y * diff_y)

    def compute_tsp(self):
        elements = [i for i in range(self.n)]
        for permutation in itertools.permutations(elements):
            new_length = 0
            for i in range(self.n - 1):
                new_length += self.each_to_each[permutation[i]][permutation[i+1]]
            new_length += self.each_to_each[permutation[self.n - 1]][permutation[0]]
            if new_length < self.shortest_tsp:
                self.shortest_tsp = new_length
                self.order_tsp = permutation
        self.order_tsp_list = list(self.order_tsp)
        self.order_tsp = tuple(self.order_tsp_list)


def convert_output(output, N):
    order = [0, 0, 0, 0, 0, 0]
    for x in range(N):
        best_y = None
        best_y_value = -1
        for y in range(N):
            if output[y*N + x] > best_y_value:
                best_y = y
                best_y_value = output[y*N + x]
        order[best_y] = x
        for i in range(N):
            output[best_y*N + i] = -1
    return order


def calculate_tsp(graph, order):
    length = 0
    for i in range(graph.n - 1):
        length += graph.each_to_each[order[i]][order[i+1]]
    length += graph.each_to_each[order[graph.n - 1]][order[0]]
    return length
