import numpy as np

node_num = 100000
edge_num = 500000
temp_name = "data/1/temp.txt"
file_name = "data/1/test_net.txt"

arr = set()

# print(arr.shape)

weight = [0] * (node_num + 1)

while len(arr) != edge_num:
    e = (np.random.randint(1, node_num + 1), np.random.randint(1, node_num + 1))
    arr.add(e)
    weight[e[1]] += 1

with open(file_name, "w") as file:
    file.write("{} {}\n".format(node_num, edge_num))
    for e in arr:
        file.write("{} {} {}\n".format(e[0], e[1], 1 / weight[e[1]]))
        # file.write("{} {}\n".format(e[0], e[1]))
