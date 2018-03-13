import csv
import numpy as np

def read_data(file_path):
    # The read-in data should be a N*W matrix,
    # where N is the length of the time sequences,
    # W is the number of sensors/data features
    i = 0
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter = ',')
        for line in reader:
            if i == 0:
                i = i+1
            else:
                line = np.array(line, dtype = 'float') # str2float
                if i == 1:
                    data = line
                else:
                    data = np.vstack((data, line))
                i += 1
    return data
