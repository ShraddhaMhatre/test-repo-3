import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv("perceptronData.csv")


##
# z-score normalization function
# takes a single feature as input and the feature mean and feature std-deviation
def zscore(feature, f_mean, f_std):
    return (feature - f_mean) / f_std


def sgn(x):
    if x > 0:
        return 1
    else:
        return -1


def perceptron(W, eta, X, y, max_itr):
    converged = False
    itr = 0
    while not converged and itr < max_itr:
        itr += 1
        flag = True
        for i in range(np.size(X, axis=0)):
            y_cap = sgn(np.sum(np.multiply(W, X[i, :])))
            if y_cap * y[i] <= 0:
                flag = False
                W = W + eta*y[i]*X[i,:]
        converged = flag

    return W


def predict(W, X):
    vectorized_sgn = np.vectorize(sgn)
    return np.apply_along_axis(vectorized_sgn, 0, np.dot(X, W[np.newaxis].T).flatten())


##
# takes a confusion matrix as input and calculates the accuracy
def calc_accuracy(cm):
    return np.sum(np.diagonal(cm)) / np.sum(cm)


def calc_precision(cm):
    true_pos = cm[-1][-1]
    return true_pos / np.sum(cm[:, -1])


def calc_recall(cm):
    true_pos = cm[-1][-1]
    return true_pos / np.sum(cm[-1, :])


def run(dataset, eta, max_itr):
    # shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # convert data frame into numpy array
    np_data = dataset.values

    accuracy_list = []
    precision_list = []
    recall_list = []

    # 10-fold
    fold_count = 0
    kf = KFold(n_splits=10)
    for train, test in kf.split(np_data):
        # split train and test data
        train_data = np_data[train]
        test_data = np_data[test]

        # separate features and labels for train data
        X_train = train_data[:, :-1]
        y_train = train_data[:, -1]

        # get the mean and std-dev for each feature of the train data
        feature_stats = stats.describe(X_train)

        # separate features and labels for test data
        X_test = test_data[:, :-1]
        y_test = test_data[:, -1]

        # z-score normalization of train and test data
        for f in range(0, np.size(X_train, 1)):
            X_train[:, f] = zscore(X_train[:, f], feature_stats.mean[f], np.sqrt(feature_stats.variance[f]))
            X_test[:, f] = zscore(X_test[:, f], feature_stats.mean[f], np.sqrt(feature_stats.variance[f]))

        # add dummy feature of ones to the train and test data
        dummy_feature_train = np.ones((np.size(X_train, 0), 1))
        dummy_feature_test = np.ones((np.size(X_test, 0), 1))
        X_train = np.append(dummy_feature_train, X_train, axis=1)
        X_test = np.append(dummy_feature_test, X_test, axis=1)

        # calculate the initial weight vector
        init_weight_vector = np.zeros(np.size(X_train, 1))

        weight_vector = perceptron(init_weight_vector, eta, X_train, y_train, max_itr)

        y_pred = predict(weight_vector, X_test)

        cm = confusion_matrix(y_test, y_pred)

        fold_accuracy = calc_accuracy(cm)
        fold_precision = calc_precision(cm)
        fold_recall = calc_recall(cm)

        accuracy_list.append(fold_accuracy)
        precision_list.append(fold_precision)
        recall_list.append(fold_recall)

    avg_accuracy = np.mean(accuracy_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)

    std_dev_accuracy = np.std(accuracy_list)
    std_dev_precision = np.std(precision_list)
    std_dev_recall = np.std(recall_list)
    print('Average Accuracy')
    print(avg_accuracy)
    print('Average Precision')
    print(avg_precision)
    print('Average Recall')
    print(avg_recall)
    print('Accuracy Standard Deviation')
    print(std_dev_accuracy)
    print('Precision Standard Deviation')
    print(std_dev_precision)
    print('Recall Standard Deviation')
    print(std_dev_recall)


run(dataset, 1, 1000)
