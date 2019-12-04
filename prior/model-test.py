import csv, sys, re
from itertools import combinations
from hamiltonian-model import runHamiltonianModel
from euclidean-model import runEuclideanModel

def enforceArguments():
    assert len(sys.argv) == 3
    subject_id = sys.argv[1]
    subject_id_w = sys.argv[2]
    assert re.match(r'^s[0]*[0-9]*$', subject_id) != None
    assert re.match(r'^s[0]*[0-9]*$', subject_id_w) != None
    assert subject_id != subject_id_w

# get subject combinations
subjects = set()
with open('../data/password-data.csv') as file:
    data = csv.reader(file, delimiter = ',')
    for row in data: subjects.add(row[0])
subjects.remove('subject')
subjects = list(subjects)

# test each combination on hamiltonian
ham_samples, euc_samples, ham_errors, euc_errors, counter = [], [], 0, 0, 0
for comb in [comb for comb in combinations(subjects, 2)]:
    print("Counter: {}".format(counter))
    counter += 1
    
    # test hamiltonian
    subject_id_h, user_score_h, subject_id_w_h, user_score_w_h = runHamiltonianModel(comb[0], comb[1])
    score_diff = user_score_w_h - user_score_h
    print("Hamiltonian {} on {}: {}".format(subject_id_w_h, subject_id_h, score_diff))
    if score_diff <= 0: ham_errors += 1
    else: ham_samples.append(score_diff)
    
    # test euclidean
    subject_id_e, user_score_e, subject_id_w_e, user_score_w_e = runEuclideanModel(comb[0], comb[1])
    score_diff = user_score_w_e - user_score_e
    print("Euclidean {} on {}: {}\n".format(subject_id_w_e, subject_id_e, score_diff))
    if score_diff <= 0: euc_errors += 1
    else: euc_samples.append(score_diff)

print("---------------------------------------")
print("Hamiltonian Mean Difference: {}".format(sum(ham_samples) / len(ham_samples)))
print("Hamiltonian Prediction Errors: {}".format(ham_errors))
print("Euclidean Mean Difference: {}".format(sum(euc_samples) / len(euc_samples)))
print("Euclidean Prediction Errors: {}".format(euc_errors))
print("---------------------------------------")