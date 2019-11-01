import collections
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def read_benchmark(benchmark_file):
    benchmarks = collections.defaultdict(list)
    with open(os.path.join(benchmark_file)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader)  # Skip header
        for row in csv_reader:
            benchmarks[tuple(row[1:5])].append((int(row[0]), float(row[6])))

    location_dict = {('50', '1'): [1],
                     ('50', '2'): [2],
                     ('50', '3'): [3],
                     ('100', '1'): [4],
                     ('100', '2'): [5],
                     ('100', '3'): [6],
                     ('200', '1'): [7],
                     ('200', '2'): [8],
                     ('200', '3'): [9]
                     }

    for param_comb in benchmarks:
        location_dict[param_comb[1], param_comb[0]].append(benchmarks[param_comb])

    for param_comb in location_dict:
        location = location_dict[param_comb][0]
        cur_subplot = plt.subplot(3, 3, location)
        for benchmark in location_dict[param_comb][1:]:
            x, y = zip(*benchmark)
            plt.plot(x, y)
        if location is 1:
            cur_subplot.legend(['ReLU dims = 50', 'ReLU dims = 100', 'ReLU dims = 200'], prop={'size':6})

        if location in range(7):
            plt.xticks([], [])
        plt.xticks(np.arange(0, 201, 50))
        if location not in [1, 4, 7]:
            plt.yticks([], [])
        plt.yticks(np.arange(0.2, 0.5, 0.05))
        if location is 4:
            plt.ylabel('F1 macro')
        if location is 8:
            plt.xlabel('Epochs')
        if location in range(4):
            title = 'LSTM layers = '.__add__(str(location))
            plt.title(title, size='8')
        if location % 3 is 0:
            ylabel = 'LSTM Dims = '.__add__(str(param_comb[0]))
            right_subplot = cur_subplot.twinx()
            right_subplot.set_ylabel(ylabel, size='8')
            right_subplot.set_yticks([], [])
        location += 1

    plt.show()
read_benchmark('NoSourceBackpropLSTM.csv')
