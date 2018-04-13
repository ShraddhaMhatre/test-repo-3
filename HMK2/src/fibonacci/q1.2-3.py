import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

##
#  read data from pdf
dataset1 = pd.read_csv("spambase.csv", header=None)
dataset2 = pd.read_csv("breastcancer.csv", header=None)
dataset3 = pd.read_csv("diabetes.csv", header=None)


##
# z-score normalization function
# takes a single feature as input and the feature mean and feature std-deviation
def zscore(feature, f_mean, f_std):
    return (feature - f_mean) / f_std


##
# calculates the sigmoid of a single value
# takes int or float
def sigmoid(x):
    return np.divide(1, 1 + np.exp(-x))


##
# W - alpha * XT.(o - y)
# o is sigmoid(WT.X)
def gradient_update_rule(W, alpha, o, X, y):
    return (W[np.newaxis].T - alpha * np.dot(X.T, (o - y[np.newaxis].T))).flatten()


##
# cross-entropy loss
# o is sigmoid(WT.X)
def loss_function(y, o):
    return (-1) * (np.dot(y, np.log(o))[0] + np.dot((1 - y), np.log(1 - o))[0])


##
# the gradient descent algorithm
def gradient_descent(W, alpha, X, y, max_iteration, tolerance):
    loss_function_list = []
    o = np.apply_along_axis(sigmoid, 0, np.dot(X, W[np.newaxis].T))
    old_loss_fun = loss_function(y, o)
    count = 0
    for i in range(0, max_iteration):
        count += 1
        W = gradient_update_rule(W, alpha, o, X, y)

        o = np.apply_along_axis(sigmoid, 0, np.dot(X, W[np.newaxis].T))

        new_loss_fun = loss_function(y, o)

        loss_function_list.append(new_loss_fun)

        if abs(old_loss_fun - new_loss_fun) <= tolerance:
            break
        else:
            old_loss_fun = new_loss_fun

    return W, loss_function_list


##
# predicts the final output
# predicts 1 if prob >= 0.5 else predicts 0
def predict(W, X_test):
    o = np.apply_along_axis(sigmoid, 0, np.dot(X_test, W[np.newaxis].T))
    return (o >= 0.5).astype(int).flatten()


##
# takes a confusion matrix as input and calculates the accuracy
def calc_accuracy(cm):
    return np.sum(np.diagonal(cm)) / np.sum(cm)


##
# takes a confusion matrix as input and calculates the precision
def calc_precision(cm):
    true_pos = np.diag(cm)
    # false_pos = np.sum(cm, axis=0) - true_pos
    return np.sum(true_pos / np.sum(cm, axis=0))


##
# takes a confusion matrix as input and calculates the recall
def calc_recall(cm):
    true_pos = np.diag(cm)
    # false_neg = np.sum(cm, axis=1) - true_pos
    return np.sum(true_pos / np.sum(cm, axis=1))


##
# driver of the algorithm
def run(dataset, dataset_name, learning_rate, max_iteration, tolerance):
    print(dataset_name, ' dataset')

    # shuffle the dataset
    dataset = dataset.sample(frac=1).reset_index(drop=True)

    # convert data frame into numpy array
    np_data = dataset.values

    # 10-fold
    fold_count = 0
    kf = KFold(n_splits=10)
    for train, test in kf.split(np_data):
        fold_count += 1
        print('fold # ', fold_count)
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

        # fit the model to the train data and calculate the final weight vector
        weight_vector, loss_function_list = gradient_descent(init_weight_vector, learning_rate, X_train, y_train, max_iteration, tolerance)

        loss_function_arr.append(loss_function_list)

        # print('Accuracy my algo: ', accuracy_score(y_test, predict(weight_vector, X_test)))
        cm = confusion_matrix(y_test, predict(weight_vector, X_test))

        fold_accuracy = calc_accuracy(cm)
        print('accuracy: ', fold_accuracy)
        fold_accuracy_list.append(fold_accuracy)

        fold_precision = calc_precision(cm)
        print('precision: ', fold_precision)
        fold_precision_list.append(fold_precision)

        fold_recall = calc_recall(cm)
        print('recall: ', fold_recall)
        fold_recall_list.append(fold_recall)

        # sklearn's model for stochastic gradient descent classifier
        # model = SGDClassifier()
        # model.fit(X_train, y_train)
        # print('Accuracy sklearn: ', accuracy_score(y_test, model.predict(X_test)))
        # print(confusion_matrix(y_test, model.predict(X_test)))


tolerance_list = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

avg_accuracy_list = []
loss_fun_list = []
for tol in tolerance_list:
    print('Tolerance Value: ', tol)
    loss_function_arr = []
    fold_accuracy_list = []
    fold_precision_list = []
    fold_recall_list = []
    run(dataset1, 'spambase', 0.00001, 1000, tol)
    print('Average accuracy: ', np.mean(fold_accuracy_list))
    avg_accuracy_list.append(np.mean(fold_accuracy_list))
    loss_fun_list.append(loss_function_arr[2][-1])
plt.plot(tolerance_list, loss_fun_list, c="r", marker='o')
plt.ylabel("loss function")
plt.xlabel("tolerance")
plt.show()

best_tol = tolerance_list[np.argmax(avg_accuracy_list)]
print('Best tolerance: ', best_tol)

loss_function_arr = []
fold_accuracy_list = []
fold_precision_list = []
fold_recall_list = []
run(dataset1, 'spambase', 0.00001, 1000, best_tol)
print('Accuracy standard deviation: ', np.std(fold_accuracy_list))
print('Average precision: ', np.mean(fold_precision_list))
print('Precision standard deviation: ', np.std(fold_precision_list))
print('Average recall: ', np.mean(fold_recall_list))
print('Recall standard deviation: ', np.std(fold_recall_list))
plt.plot(np.arange(0, len(loss_function_arr[2])), loss_function_arr[2], c="r", marker='o')
plt.ylabel("loss function")
plt.xlabel("iterations")
plt.show()

avg_accuracy_list = []
loss_fun_list = []
loss_function_arr = []
for tol in tolerance_list:
    loss_function_arr = []
    fold_accuracy_list = []
    fold_precision_list = []
    fold_recall_list = []
    run(dataset2, 'breast cancer', 0.00001, 1000, tol)
    avg_accuracy_list.append(np.mean(fold_accuracy_list))
    loss_fun_list.append(loss_function_arr[2][-1])
plt.plot(tolerance_list, loss_fun_list, c="r", marker='o')
plt.ylabel("loss function")
plt.xlabel("tolerance")
plt.show()

best_tol = tolerance_list[np.argmax(avg_accuracy_list)]
print('Best tolerance: ', best_tol)

loss_function_arr = []
fold_accuracy_list = []
fold_precision_list = []
fold_recall_list = []
run(dataset2, 'breast cancer', 0.00001, 1000, best_tol)
print('Average accuracy: ', np.mean(fold_accuracy_list))
print('Accuracy standard deviation: ', np.std(fold_accuracy_list))
print('Average precision: ', np.mean(fold_precision_list))
print('Precision standard deviation: ', np.std(fold_precision_list))
print('Average recall: ', np.mean(fold_recall_list))
print('Recall standard deviation: ', np.std(fold_recall_list))
plt.plot(np.arange(0, len(loss_function_arr[2])), loss_function_arr[2], c="r", marker='o')
plt.ylabel("loss function")
plt.xlabel("iterations")
plt.show()

avg_accuracy_list = []
loss_fun_list = []
loss_function_arr = []
for tol in tolerance_list:
    loss_function_arr = []
    fold_accuracy_list = []
    fold_precision_list = []
    fold_recall_list = []
    run(dataset3, 'diabetes', 0.00001, 1000, tol)
    avg_accuracy_list.append(np.mean(fold_accuracy_list))
    loss_fun_list.append(loss_function_arr[2][-1])
plt.plot(tolerance_list, loss_fun_list, c="r", marker='o')
plt.ylabel("loss function")
plt.xlabel("tolerance")
plt.show()

best_tol = tolerance_list[np.argmax(avg_accuracy_list)]
print('Best tolerance: ', best_tol)

loss_function_arr = []
fold_accuracy_list = []
fold_precision_list = []
fold_recall_list = []
run(dataset3, 'diabetes', 0.00001, 1000, best_tol)
print('Average accuracy: ', np.mean(fold_accuracy_list))
print('Accuracy standard deviation: ', np.std(fold_accuracy_list))
print('Average precision: ', np.mean(fold_precision_list))
print('Precision standard deviation: ', np.std(fold_precision_list))
print('Average recall: ', np.mean(fold_recall_list))
print('Recall standard deviation: ', np.std(fold_recall_list))
plt.plot(np.arange(0, len(loss_function_arr[2])), loss_function_arr[2], c="r", marker='o')
plt.ylabel("loss function")
plt.xlabel("iterations")
plt.show()
