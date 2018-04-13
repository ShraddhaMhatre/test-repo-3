import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
from numpy.linalg import inv
import matplotlib.pyplot as plt

# read data from pdf
dataset1 = pd.read_csv("yachtData.csv", header=None)
# dataset2 = pd.read_csv("yatchData.csv", header=None)
# dataset3 = pd.read_csv("concreteData.csv", header=None)


def calc_SSE(w, x, y):
    return np.sum(np.square(np.dot(x, w) - y[np.newaxis].T))


def calc_RMSE(w, x, y):
    return np.sqrt(calc_SSE(w, x, y) / np.size(x, 0))


def zscore(feature, f_mean, f_std):
    return (feature - f_mean) / f_std


def gradient_descent(x, y):
    return np.dot(np.dot(inv(np.dot(x.T, x)), x.T), y[np.newaxis].T)


def run(dataset, p):
    # shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # convert dataframe into numpy array
    np_data = dataset.values

    # 10-fold
    kf = KFold(n_splits=10)

    for train, test in kf.split(np_data):
        train_data = np_data[train]
        test_data = np_data[test]

        feature_train = train_data[:, :-1]
        class_train = train_data[:, -1]

        feature_test = test_data[:, :-1]
        class_test = test_data[:, -1]

        expanded_feature_train = feature_train
        expanded_feature_test = feature_test

        if p > 1:
            for power in range(2, p + 1):
                for f in range(np.size(feature_train, 1)):
                    expanded_feature_train = np.column_stack((expanded_feature_train, np.power(feature_train[:, f], power)))
                    expanded_feature_test = np.column_stack((expanded_feature_test, np.power(feature_test[:, f], power)))

        expanded_feature_stats = stats.describe(expanded_feature_train)

        # z score
        for f in range(0, np.size(expanded_feature_train, 1)):
            expanded_feature_train[:, f] = zscore(expanded_feature_train[:, f], expanded_feature_stats.mean[f], np.sqrt(expanded_feature_stats.variance[f]))
            expanded_feature_test[:, f] = zscore(expanded_feature_test[:, f], expanded_feature_stats.mean[f], np.sqrt(expanded_feature_stats.variance[f]))

        add_feature_train = np.ones((np.size(expanded_feature_train, 0), 1))
        add_feature_test = np.ones((np.size(expanded_feature_test, 0), 1))

        feature_train = np.append(add_feature_train, expanded_feature_train, axis=1)
        feature_test = np.append(add_feature_test, expanded_feature_test, axis=1)

        init_w = np.zeros((np.size(feature_train, 1), 1))
        weight_vector = gradient_descent(feature_train, class_train)

        train_RMSE = calc_RMSE(weight_vector, feature_train, class_train)
        train_RMSE_arr.append(train_RMSE)

        test_RMSE = calc_RMSE(weight_vector, feature_test, class_test)
        test_RMSE_arr.append(test_RMSE)

    avg_train_RMSE_arr.append(np.average(train_RMSE_arr) / 10)
    avg_test_RMSE_arr.append(np.average(test_RMSE_arr) / 10)


max_p = 7
train_RMSE_arr = []
test_RMSE_arr = []
avg_train_RMSE_arr = []
avg_test_RMSE_arr = []

for po in range(1, max_p + 1):
    run(dataset1, po)

# run(dataset2)
# run(dataset3)
print(avg_train_RMSE_arr)
print(avg_test_RMSE_arr)
plt.scatter(np.arange(1, max_p + 1), avg_train_RMSE_arr, c="r")
plt.scatter(np.arange(1, max_p + 1), avg_test_RMSE_arr, c="b")
plt.ylabel("Average RMSE")
plt.xlabel("powers")
plt.show()
