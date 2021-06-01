import numpy as np
import os
import torch
import matplotlib.pyplot as plt

Super_element_dim = 294
Super_step_size = 1550

earthquake_data_dir = "./data/EL_Centro_NS.txt"

local_path = "./data/"
remote_path = ""

def Super_get_res(earthquake_id, element_dim=294):
    import pickle
    print("Super Loading  earthquake :" + str(earthquake_id))
    path = local_path + "/res_dict_" + "0"*(3-len(str(earthquake_id))) +str(earthquake_id) + ".pkl"
    data = pickle.load(open(path, 'rb'))
    nodeid_list = [int(i.split("@")[-2].split('.')[-1]) for i in list(data.keys())]
    res = []
    
    for node_id in nodeid_list:
        index = '@'.join(["00" + str(earthquake_id), "Step-1", "Node PART-1-1."+str(node_id), "A1"])
        step = int(0.02 / (data[index][1][0] - data[index][0][0]))
        res.append([data[index][i][-1] for i in range(0, Super_step_size*int(step), int(step))])

    return np.array(res).T


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
        result[node] = earthquake_data

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

    data = transformer(np.array(data))

    earthquake = get_earthquake_data()

    data_acc = torch.tensor(data).to(torch.float32)
    data_earthquake = torch.tensor(earthquake).reshape(len(earthquake), 1).to(torch.float32)
    data = torch.cat((data_acc, data_earthquake), dim=1)

    return data.unsqueeze(dim=1)

def get_from_pickle(size=31):
    import pickle
    #path_data = "./data/res_dict.pkl"
    path = "./data/id_data_map.pkl"
    #path_earthquake = "./data/id_wavename_map.pkl"

    #res_dict = pickle.load(open(path_data, 'rb'))
    id_data_map = pickle.load(open(path, 'rb'))
    #id_wavename_map = pickle.load(open(path_earthquake, 'rb'))

    """
    def get_res_(earthquake_id, node_id):
        return np.array([i[-1] for i in res_dict['@'.join([str(earthquake_id), 'Step-1', 'Node PART-1-1.' + str(node_id), 'A1'])]])[:Super_step_size]

    def get_res(earthquake_id):
        res = []
        for i in range(1, Super_element_dim+1):
            res.append(get_res_(earthquake_id, i))
        return np.array(res).T
    """

    def extract(a):
        step = 0.02 / (a[1][0] - a[0][0])
        return np.array([[a[i][-1] for i in range(0, Super_step_size*int(step), int(step))]]).T

    raw_data = []

    for i in range(size):
        a = id_data_map[i]
        step = 0.02 / (a[1][0] - a[0][0])
        if Super_step_size * int(step) > len(a):
            print("Earthquake " + str(i) + " is too short, pass ")
            continue
        earthquake = extract(id_data_map[i])
        
        tmp = Super_get_res(i)

        print(tmp.shape)

        raw_data.append(np.concatenate((tmp, earthquake), axis=1))

    return torch.tensor(raw_data).reshape(-1, Super_step_size, 1, Super_element_dim+1)


def get_dataset():
    #data = torch.cat((get_from_pickle(), task().reshape(-1, Super_step_size, 1, Super_element_dim+1)), dim=0)
    data = get_from_pickle()
    return data.to(torch.float32)

if __name__ == "__main__":
    #tmp = task()
    #tmp = get_from_pickle()
    tmp = get_dataset()
    print(tmp.shape)