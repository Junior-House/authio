import csv, sys
import numpy as np
import re

def runHamiltonianModel(subject_id, subject_id_w):

    # collect subject data for correct user
    seen_data = False
    subject_data = np.array([[0, 0], [0, 0]])
    with open('../data/password-data.csv') as file:
        data = csv.reader(file, delimiter = ',')
        for row in data:
            if row[0] == 'subject': continue
            if row[0] != subject_id and seen_data == True: break
            elif row[0] != subject_id and seen_data == False: continue
            elif seen_data == False: seen_data = True
            num_row = [float(d) for d in row[3:]]
            new_data = np.array(num_row)
            if subject_data.shape == (2, 2): subject_data = new_data
            else: subject_data = np.vstack([subject_data, new_data])
    assert np.sum(subject_data) > 0.

    # collect subject data for wrong user
    seen_data = False
    subject_data_w = np.array([[0, 0], [0, 0]])
    with open('../data/password-data.csv') as file:
        data = csv.reader(file, delimiter = ',')
        for row in data:
            if row[0] == 'subject': continue
            if row[0] != subject_id_w and seen_data == True: break
            elif row[0] != subject_id_w and seen_data == False: continue
            elif seen_data == False: seen_data = True
            num_row = [float(d) for d in row[3:]]
            new_data = np.array(num_row)
            if subject_data_w.shape == (2, 2): subject_data_w = new_data
            else: subject_data_w = np.vstack([subject_data_w, new_data])
    assert np.sum(subject_data_w) > 0.

    # partition data for correct user
    data_partition = (0.90, 0.10)
    assert sum(data_partition) == 1.0
    dimensions = subject_data.shape
    train_size = int(round(dimensions[0] * data_partition[0]))
    test_size = dimensions[0] - train_size

    # train data for correct user
    mean_vector = [0] * dimensions[1]
    for i in range(train_size):
        mean_vector = np.add(mean_vector, subject_data[i,:])
    mean_vector = np.multiply(1. / float(train_size), mean_vector)

    # test data on correct user
    scores = []
    for i in range(test_size):
        j = i + train_size
        score = np.subtract(mean_vector, subject_data[j,:])
        score = np.absolute(score)
        scores.append(np.sum(score))
    user_score = sum(scores) / len(scores)

    # test data on wrong user
    scores = []
    for i in range(subject_data_w.shape[0]):
        score = np.subtract(mean_vector, subject_data_w[i,:])
        score = np.absolute(score)
        scores.append(np.sum(score))
    user_score_w = sum(scores) / len(scores)

    return (subject_id, user_score, subject_id_w, user_score_w)
