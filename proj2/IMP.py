import argparse
import math
import sys
import time

import numpy as np

memory_limit = 1600


def read_file(network_path):
    file = open(network_path)
    temp = file.readline()
    n, m = [int(x) for x in temp.split(" ")]

    edges = []
    for _ in range(n + 1):
        edges.append([])

    for _ in range(m):
        temp = file.readline().split(" ")
        u, v, w = int(temp[0]), int(temp[1]), float(temp[2])
        edges[v].append((u, w))

    return edges


def log(x):
    return math.log(x, math.e)


class RRS:
    def __init__(self, edges, model):
        self.edges = edges
        self.model = model
        self.rrs = []
        self.seed = []
        self.memory = 0
        for e in edges:
            self.memory += sys.getsizeof(e) / 1000000
        # print(self.memory)

    def output(self, file_name):
        file = open(file_name)
        for e in self.seed:
            file.write(e + "\n")

    def addRR_LT(self):
        node = np.random.randint(1, len(self.edges))
        rr, active = [node], [node]

        while len(active):
            e = active.pop(0)
            if len(self.edges[e]) == 0:
                continue
            edge = edges[e][np.random.randint(len(edges[e]))]
            if edge[0] in rr:
                continue

            active.append(edge[0])
            rr.append(edge[0])
        self.rrs.append(rr)
        self.memory += sys.getsizeof(rr) / 1000000

    def addRR_IC(self):
        node = int(np.random.randint(1, len(self.edges)))
        rr, active = [node], [node]

        while len(active):
            e = active.pop(0)
            for edge in self.edges[e]:
                if edge[0] in rr:
                    continue

                if np.random.rand() < edge[1]:
                    active.append(edge[0])
                    rr.append(edge[0])
        self.rrs.append(rr)
        self.memory += sys.getsizeof(rr) / 1000000

    def addRR(self):
        if self.model == 'LT':
            self.addRR_LT()
        else:
            self.addRR_IC()

    def generate(self, k, e, l):
        n = len(self.edges) - 1
        lb = 1
        e_ = math.sqrt(2) * e
        lnk = sum(log(x) for x in range(n - k + 1, n + 1)) - sum(log(x) for x in range(1, k + 1))
        lamda = (2 + 2 / 3 * e_) * (lnk + l * log(n) + log(math.log(n, 2))) * n / math.pow(e_, 2)

        # get the time and memory for every 10000 rr
        rr_generate_time, rr_generate_memory = time.time(), self.memory
        while len(self.rrs) < 50000:
            self.addRR()
        rr_generate_time, rr_generate_memory = (time.time() - rr_generate_time) / 5, \
                                               (self.memory - rr_generate_memory) / 5
        self.seed, v, select_time = self.select([k])

        for i in range(1, int(math.log(n, 2))):
            x = n / math.pow(2, i)
            theta = lamda / x

            drr = theta - len(self.rrs)
            need_time = time.time() - start + drr * rr_generate_time / 10000
            need_memory = self.memory + drr * rr_generate_memory / 10000
            # if source is not enough
            if need_time > time_limit - theta / len(self.rrs) * select_time or need_memory > memory_limit:
                # print("RRRRRRRRR")
                roundt = int((time_limit - theta / len(self.rrs) * select_time - (time.time() - start)) / rr_generate_time)
                roundm = int((memory_limit - self.memory) / rr_generate_memory)
                round = min(roundt, roundm)
                for _ in range(round * 10000):
                    self.addRR()
                self.seed, v, select_time = self.select([k])
                return

            while len(self.rrs) <= theta:
                self.addRR()

            self.seed, v, select_time = self.select([k])
            if n * v >= (1 + e_) * x:
                lb = n * v / (1 + e_)
                break

        alpha = math.sqrt(l * log(n) + log(2))
        beta = math.sqrt((1 - 1 / math.e) * (lnk + l * log(n) + log(2)))
        lamda = 2 * n * math.pow((1 - 1 / math.e) * alpha + beta, 2) * math.pow(e, -2)
        theta = lamda / lb

        drr = theta - len(self.rrs)
        if drr * rr_generate_time / 10000 + time.time() - start + select_time > time_limit \
                or drr * rr_generate_memory / 10000 + self.memory > memory_limit:
            # print("FFFFFFFF!")
            return
        while len(self.rrs) < theta:
            self.addRR()

        if time.time() - start > time_limit - select_time * 1.5:
            # print("FFFFFFFF!")
            return
        self.seed, v, select_time = rrs.select([k])

    def select(self, arr):
        result = []

        count = [0] * len(self.edges)
        at = {}
        crr = 0

        select_start = time.time()
        for i in range(len(self.rrs)):
            for j in self.rrs[i]:
                count[j] += 1
                if j in at.keys():
                    at[j].append(i)
                else:
                    at[j] = [i]

        for k in arr:
            while len(result) < k:
                idx = count.index(max(count))
                result.append(idx)
                temp = at[idx][:]
                for i in temp:
                    crr += 1
                    rr = self.rrs[i]
                    for j in rr:
                        count[j] -= 1
                        at[j].remove(i)

            # print(len(self.rrs), self.memory, " : ", time.time() - select_start)

        v = crr / len(self.rrs)
        # print(result, v)
        return result, v, time.time() - select_start


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='data/1/NetHEPT.txt')
    parser.add_argument('-k', '--seed_count', type=int, default=500)
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=120)

    args = parser.parse_args()
    file_name = args.file_name
    k = args.seed_count
    model = args.model
    time_limit = args.time_limit - 3

    edges = read_file(file_name)

    rrs = RRS(edges, model)
    rrs.generate(k=k, e=0.1, l=1 * (1 + log(2) / log(len(edges) - 1)))

    for e in rrs.seed:
        print(e)

    sys.stdout.flush()

    # end = time.time()
    # print(end - start)
