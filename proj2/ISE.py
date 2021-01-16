import numpy as np
import time
import multiprocessing as mp
import sys
import argparse
import queue

core = 8

def read_file(network_path, seeds_path):
    file = open(network_path)
    temp = file.readline()
    n, m = [int(e) for e in temp.split(" ")]

    edges = []
    for _ in range(n + 1):
        edges.append([])

    for _ in range(m):
        temp = file.readline().split(" ")
        i, j, w = int(temp[0]), int(temp[1]), float(temp[2])
        edges[i].append((j, w))

    # for i in range(len(edges)):
    #     edges[i] = set(edges[i])
        # show(nodes)

    file = open(seeds_path)
    activated = []
    while 1:
        temp = file.readline()
        if not temp:
            break
        activated.append(int(temp))

    return edges, set(activated)


class ISE:
    def __init__(self, graph_name, seed_name, model, tl, start, cnt):
        self.edges, self.seed = read_file(graph_name, seed_name)
        self.model, self.tl, self.cnt, self.start = model, tl, cnt, start

    # def ISE_IC(self):
    #     result = []
    #
    #     for i in range(self.cnt):
    #         # initialization
    #         activated, active, active_num = queue.SimpleQueue(), [0] * len(self.edges), 0
    #         for e in self.seed:
    #             activated.put(e)
    #             active[e] = 1
    #             active_num += 1
    #
    #         while not activated.empty():
    #             e = activated.get()
    #             for y, w in self.edges[e]:
    #                 if active[y]:
    #                     continue
    #
    #                 v = np.random.random()
    #                 if v <= w:
    #                     active[y] = 1
    #                     activated.put(y)
    #                     active_num += 1
    #
    #         result.append(active_num)
    #         end = time.time()
    #         if end - self.start > self.tl - 5:
    #             break
    #
    #     return sum(result) / len(result)

    def ISE_IC(self):
        result = []

        for i in range(self.cnt):
            # initialization
            activated, active = [], [0] * len(self.edges)
            for e in self.seed:
                activated.append(e)
                active[e] = 1

            for e in activated:
                for y, w in self.edges[e]:
                    if active[y]:
                        continue

                    v = np.random.random()
                    if v <= w:
                        active[y] = 1
                        activated.append(y)

            result.append(len(activated))
            end = time.time()
            if end - self.start > self.tl - 5:
                break

        return sum(result) / len(result)

    def ISE_LT(self):
        result = []

        for i in range(self.cnt):
            # initialization
            activated, active, threshold = [], [0] * len(self.edges), np.random.random(len(self.edges))
            for e in self.seed:
                activated.append(e)
                active[e] = 1

            for e in activated:
                # x = activated.pop()
                for y, w in self.edges[e]:
                    if active[y]:
                        continue

                    if threshold[y] <= w:
                        active[y] = 1
                        activated.append(y)
                    else:
                        threshold[y] -= w

            result.append(len(activated))
            end = time.time()
            if end - self.start > self.tl - 5:
                break

        return sum(result) / len(result)

    def evaluate_seed(self):
        result = []
        pool = mp.Pool(core)
        if self.model == "IC":
            for i in range(core):
                result.append(pool.apply_async(self.ISE_IC).get())
        else:
            for i in range(core):
                result.append(pool.apply_async(self.ISE_LT).get())
        pool.close()
        pool.join()
        return sum(result) / len(result)


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='data/1/NetHEPT.txt')
    parser.add_argument('-s', '--seed', type=str, default='data/1/test_seed.txt')
    parser.add_argument('-m', '--model', type=str, default='IC')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = args.file_name
    seed = args.seed
    model = args.model
    time_limit = args.time_limit
    ai = ISE(file_name, seed, model, time_limit, start, 300)

    np.random.seed(int(start))
    test = time.time()
    print(ai.evaluate_seed())

    sys.stdout.flush()

    end = time.time()
    print(end - start)
