import argparse
import numpy as np


def calculate_acc(filepath,epochs,max_fold):
    
    validation_loss = np.zeros((epochs, max_fold))
    test_accuracy = np.zeros((epochs, max_fold))
    test_acc = np.zeros(max_fold)
    with open(filepath, 'r') as filehandle:
        filecontents = filehandle.readlines()
        index = 0
        col = 0
        for line in filecontents:
            ss = line.split()
            t_acc = ss[1]
            v_loss = ss[0]
            validation_loss[index][col] = float(v_loss)
            test_accuracy[index][col] = float(t_acc)
            index += 1
            if index == epochs:
                index = 0
                col += 1
                if col == max_fold:
                    break

    min_ind = np.argmin(validation_loss, axis=0)
    for i in range(max_fold):
        ind = min_ind[i]
        test_acc[i] = test_accuracy[ind][i]
    ave_acc = np.mean(test_acc)
    std_acc = np.std(test_acc)
    return ave_acc

