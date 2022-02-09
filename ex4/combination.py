import numpy as np
from ex4 import calc_accuracy
from sklearn.metrics import classification_report
from ex4 import log_results


test_y_1 = np.loadtxt("test_y_88.2", dtype='uint8').tolist()
test_y_2 = np.loadtxt("test_y_svm_88.68", dtype='uint8').tolist()

labels = np.loadtxt("test_labels", dtype='uint8')
union = []
for index, pred in enumerate(test_y_1):
    if test_y_2[index] in [0, 5, 2]:
        union.append(test_y_2[index])
    elif test_y_1[index] in [1, 3, 8, 9]:
        union.append(test_y_1[index])
    else:
        union.append(test_y_2[index])


test_y = np.loadtxt("test_y", dtype='uint8').tolist()

print(classification_report(labels, test_y_1))
print(classification_report(labels, test_y_2))
print(calc_accuracy(test_y, labels))

log_results(union, filename="test_y")
