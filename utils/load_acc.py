import numpy as np

Super_element_dim = 153
Super_step_size = 101


def task():
    path = "../data/acc.txt"
    with open(path, 'r') as f:
        raw = f.readlines()

    data = []
    for e in raw:
        tmp = e.split('\t')
        data.append(float(tmp[1].split('\n')[0]))

    return np.array(data)


def transformer(data):
    result = []
    for i in range(0, Super_step_size*Super_element_dim, Super_step_size):
        result.append(data[i:i+Super_step_size])
    
    result = np.array(result).T

    print(result.shape)

    return result


if __name__ == "__main__":
    tmp = task()
    result = transformer(tmp)
    print(result[0].shape)