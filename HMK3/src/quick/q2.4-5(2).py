import numpy as np
import operator
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from decimal import getcontext, Decimal

getcontext().prec = 30


##
# takes a confusion matrix as input and calculates the accuracy
def calc_accuracy(cm):
    return np.sum(np.diagonal(cm)) / np.sum(cm)


##
# takes a confusion matrix as input and calculates the precision
def calc_precision(cm):
    true_pos = np.diag(cm)
    return true_pos / np.sum(cm, axis=0)


##
# takes a confusion matrix as input and calculates the recall
def calc_recall(cm):
    true_pos = np.diag(cm)
    return true_pos / np.sum(cm, axis=1)


f = open('train.data', 'r')

##
# key: word_id
# value: word frequency across all documents
word_frequency_dict = dict()

##
# key: doc_id
# value: list of word_ids appearing in that document
doc_word_map = dict()
doc_word_freq_map = dict()

##
# Number of documents in the training set
N = 0
for line in f.readlines():
    doc_id = int(line.split()[0])
    word_id = int(line.split()[1])
    count = int(line.split()[2])

    if doc_id > N:
        N = doc_id

    if doc_id in doc_word_freq_map:
        doc_word_freq_map[doc_id].append((word_id, count))
    else:
        doc_word_freq_map[doc_id] = [(word_id, count)]

    if doc_id in doc_word_map:
        doc_word_map[doc_id].append(word_id)
    else:
        doc_word_map[doc_id] = [word_id]

    if word_id in word_frequency_dict:
        word_frequency_dict[word_id] += count
    else:
        word_frequency_dict[word_id] = count

f.close()
#
# print(doc_word_freq_map)

##
# List of tuples (word_id, frequency)
# sorted in descending order according to the frequency
sorted_word_frequency = sorted(word_frequency_dict.items(), key=operator.itemgetter(1), reverse=True)
# print(sorted_word_frequency)

# vocabulary_sizes = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, len(sorted_word_frequency)]
# print(vocabulary_sizes)
vocabulary_sizes = [100, 150, 200, 250]

accuracy_list = []
for V in vocabulary_sizes:
    X = np.zeros((N, 100))

    for i in range(100):
        word_id = sorted_word_frequency[i][0]
        for key, value in doc_word_freq_map.items():
            for tup in value:
                if word_id in tup:
                    X[key - 1][i] = tup[1]
                    break
    print('X matrix:')
    print(X)

    f = open('train.label', 'r')

    labels = list()
    for line in f.readlines():
        labels.append(int(line))
    # print(labels)

    f.close()

    y = np.zeros((N, 20))
    doc_id = 0
    for class_id in labels:
        y[doc_id][class_id - 1] = 1
        doc_id += 1
    print('y matrix:')
    print(y)

    f = open('test.data', 'r')

    doc_word_map_test = dict()
    doc_word_freq_map_test = dict()
    N_test = 0
    for line in f.readlines():
        doc_id = int(line.split()[0])
        word_id = int(line.split()[1])
        count = int(line.split()[2])

        if doc_id > N_test:
            N_test = doc_id

        if doc_id in doc_word_freq_map_test:
            doc_word_freq_map_test[doc_id].append((word_id, count))
        else:
            doc_word_freq_map_test[doc_id] = [(word_id, count)]

        if doc_id in doc_word_map_test:
            doc_word_map_test[doc_id].append(word_id)
        else:
            doc_word_map_test[doc_id] = [word_id]

    f.close()

    # print(doc_word_map_test)
    # print(doc_word_freq_map_test)

    X_test = np.zeros((N_test, 100))
    for i in range(100):
        word_id = sorted_word_frequency[i][0]
        for key, value in doc_word_freq_map_test.items():
            for tup in value:
                if word_id in tup:
                    X_test[key - 1][i] = tup[1]
                    break
    print('X_test matrix:')
    print(X_test)

    theta_matrix = np.zeros((100, 20))
    pi_matrix = np.zeros(20)
    for k in range(20):
        for j in range(100):
            numerator_jk = np.sum(np.multiply(y[:, k], X[:, j])) + 1
            denominator_ik = np.sum(np.multiply(y[:, k], np.sum(X, axis=1))) + 2

            theta_matrix[j][k] = numerator_jk / denominator_ik

    pi_matrix = np.sum(y, axis=0)

    print('theta matrix:')
    print(theta_matrix)
    print(theta_matrix.shape)
    print('pi matrix:')
    print(pi_matrix)
    print(pi_matrix.shape)

    print('N_test:')
    print(N_test)
    prediction_matrix = np.zeros((N_test, 20))
    predicted = []
    for i in range(N_test):
        denominator = 0
        numerator_list = []
        for k in range(20):
            numerator = 1
            for j in range(100):
                numerator *= math.pow(theta_matrix[j][k], X_test[i][j])
            numerator_list.append(numerator * pi_matrix[k])
            denominator += numerator * pi_matrix[k]

        class_probability_list = []

        if denominator == 0.0:
            print('denominator:')
            print(denominator)

        for n in numerator_list:
            class_probability_list.append(n / denominator)

        predicted.append(np.argmax(class_probability_list) + 1)
        prediction_matrix[i][np.argmax(class_probability_list)] = 1
    print('predicted:')
    print(predicted)
    #
    # # print(prediction_matrix)
    #
    # # predicted = []
    # # for i in range(N_test):
    # #     predicted.append(np.nonzero(prediction_matrix[i:])[0] + 1)
    # # print(predicted)
    #
    f = open('test.label', 'r')

    actual = []
    for line in f.readlines():
        actual.append(int(line))

    f.close()
    print('actual:')
    print(actual)

    confusion_mat = confusion_matrix(actual, predicted)
    # print(accuracy_score(actual, predicted))

    accuracy_list.append(calc_accuracy(confusion_mat))
    print('precision', calc_precision(confusion_mat))
    print('recall', calc_recall(confusion_mat))

plt.plot(vocabulary_sizes, accuracy_list, c="r", marker='o')
plt.ylabel("Accuracy")
plt.xlabel("Vocabulary Size")
plt.show()
