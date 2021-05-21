import numpy as np
import os

import matplotlib.pyplot as plt

Super_element_dim = 153
Super_step_size = 1550

earthquake_data_dir = "./data/EL_Centro_NS.txt"

def get_earthquake_data():
    with open(earthquake_data_dir, 'r') as f:
        e_d_file = [float(e.split('\n')[0]) for e in f.readlines()]

    earthquake_data = np.array(e_d_file[:Super_step_size])

    return earthquake_data

def transformer(data):
    result = []
    for i in range(0, Super_step_size*Super_element_dim, Super_step_size):
        result.append(data[i:i+Super_step_size])
    
    result = np.array(result)

    print(result.shape)

    earthquake_data = get_earthquake_data()

    node_list = range(1, 37, 3)

    for node in node_list:
        #print(node)
        result[node] = earthquake_data
    
    #print(result[1])

    result = result.T

    print(result.shape)

    return result

def task():
    path = "./data/new.txt"
    with open(path, 'r') as f:
        raw = f.readlines()

    data = []
    for e in raw:
        tmp = e.split('\t')
        data.append(float(tmp[1].split('\n')[0]))

    print(len(data))

    return transformer(np.array(data))


if __name__ == "__main__":
    tmp = task()
    print(tmp[0].shape)