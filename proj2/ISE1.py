import multiprocessing as mp
import time
import sys
import argparse
import os
import numpy as np
import copy
import time

core = 8
start = time.time()

def read_file(network_path, seeds_path):
    file = open(network_path)
    temp = file.readline()
    n, m = [int(e) for e in temp.split(" ")]
    # print(n, m)

    edges = []
    for i in range(n + 1):
        edges.append([])

    for i in range(m):
        temp = file.readline().split(" ")
        i, j, w = int(temp[0]), int(temp[1]), float(temp[2])
        edges[i].append((j, w))

        # show(nodes)

    file = open(seeds_path)
    activated = []
    while 1:
        temp = file.readline()
        if not temp:
            break
        activated.append(int(temp))

    return edges, activated, len(activated)


def ISE_IC(edges, initial_activated, initial_active, tl):
    result, active = [], [0]*len(edges)

    for i in range(200):
        # initialization
        activated, active_num = copy.deepcopy(initial_activated), initial_active
        for i in range(len(active)):
            active[i] = 0
        for e in activated:
            active[e] = 1

        for e in activated:
            # x = activated.pop()
            for y, w in edges[e]:
                if active[y]:
                    continue

                v = np.random.random()
                if v <= w:
                    active_num += 1
                    active[y] = 1
                    activated.append(y)

        result.append(active_num)
        end = time.time()
        if end - start > tl - 5:
            break

    return sum(result) / len(result)


def ISE_LT(edges, initial_activated, initial_active, tl):
    result, active, threshold = [], [0]*len(edges), [0]*len(edges)

    for i in range(200):
        # initialization
        activated, active_num = copy.deepcopy(initial_activated), initial_active
        for i in range(len(active)):
            active[i] = 0
            threshold[i] = np.random.random()
        for e in activated:
            active[e] = 1

        for e in activated:
            # x = activated.pop()
            for y, w in edges[e]:
                if active[y]:
                    continue

                threshold[y] -= w
                if threshold[y] <= 0:
                    active_num += 1
                    active[y] = 1
                    activated.append(y)

        result.append(active_num)
        end = time.time()
        if end - start > tl - 5:
            break

    return sum(result) / len(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--file_name', type=str, default='data/1/test_net.txt')
    parser.add_argument('-s', '--seed', type=str, default='data/1/test_seed.txt')
    parser.add_argument('-m', '--model', type=str, default='LT')
    parser.add_argument('-t', '--time_limit', type=int, default=60)

    args = parser.parse_args()
    file_name = args.file_name
    seed = args.seed
    model = args.model
    time_limit = args.time_limit

    # print(file_name, seed, model, time_limit)

    edges, activated, active = read_file(file_name, seed)
    np.random.seed(int(start))
    pool = mp.Pool(core)
    result = []

    if model == "IC":
        for i in range(core):
            result.append(pool.apply_async(ISE_IC, args=(edges, activated, active, time_limit)).get())
    else:
        for i in range(core):
            result.append(pool.apply_async(ISE_LT, args=(edges, activated, active, time_limit)).get())

    pool.close()
    pool.join()

    # print(result)
    print(sum(result) / 8)

    print(time.time()-start)
    sys.stdout.flush()
