#! /usr/bin/python3

import os

file_path = "./fem_data.txt"

def read_file(path):
    with open(path, 'r') as f:
        lines = f.read()
    return lines

def run():
    f = read_file(file_path).split('\n\n')
    
    for i in range(10):
        print(f[i])
        print("--------------")

    data = [e.split('\n') for e in f]

    import re

    bag = [element for element in data if 'ROW' in element[0] and 'MATRIX' in element[0]]
    
    print(bag[0])

    


if __name__ == "__main__":
    run()
