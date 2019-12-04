from data.data import buildData
from model.model import runModel
import numpy as np
import time
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try: yield
        finally: 
            sys.stdout = old_stdout
            sys.stderr = old_stderr

if __name__ == "__main__":
    verbose = False

    # take start time
    start = int(round(time.time() * 1000))

    # test each user as valid
    results = []
    for user in range(51):
        print('Testing for user: {}'.format(user))
        with suppress_stdout():
            buildData(user)
            result = list(runModel())
            results.append(result)

        # print interim results
        if verbose:
            print("Accuracy: {}".format(result[0]))
            print("Recall: {}".format(result[1]))
            print("Precision: {}".format(result[2]))
            print("F1: {}".format(result[3]))

    # calculate final results
    results = np.array(results, dtype=np.float64)
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)

    # print final results
    print('\nFinal accuracy mean: {}'.format(means[0]))
    print('Final accuracy std: {}'.format(stds[0]))
    print('\nFinal recall mean: {}'.format(means[1]))
    print('Final recall std: {}'.format(stds[1]))
    print('\nFinal precision mean: {}'.format(means[2]))
    print('Final precision std: {}'.format(stds[2]))
    print('\nFinal F1 mean: {}'.format(means[3]))
    print('Final F1 std: {}'.format(stds[3]))
    
    # calculate total time
    end = int(round(time.time() * 1000))
    print('\nTotal time expended: {}'.format(end - start))
