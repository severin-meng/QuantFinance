"""
Exercise 2.8 - Compare the least squares vs kNN performance for the zipcode dataset classification.
The dataset can be found here: https://hastie.su.domains/ElemStatLearn/
"""

import os
import numpy as np
from sklearn import linear_model, preprocessing, neighbors

base_path = r"C:\Quantitative Finance\Elements of Statistical Learning\zipcode dataset"
train_data = os.path.join(base_path, "zip.train")
test_data = os.path.join(base_path, "zip.test")


def fit_linear_model(X_train, y_train, X_test, y_test, penalty=None, scale=False):
    lin = linear_model.LogisticRegression(penalty=penalty)
    lin.n_iter_ = 500
    if scale:
        scaler = preprocessing.StandardScaler().fit(X_train)  # scale data - otherwise, penalized algorithm does not converge
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # must not fit scaler again!
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    lin.fit(X_train_scaled, y_train)
    print(f"Linear (penalty={penalty}) training performance: {lin.score(X_train_scaled, y_train)}")
    print(f"Linear (penalty={penalty}) test performance: {lin.score(X_test_scaled, y_test)}")


def fit_knn(X_train, y_train, X_test, y_test, k):
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(f"KNN (k={k}) training performance: {knn.score(X_train, y_train)}")
    print(f"KNN (k={k}) test performance: {knn.score(X_test, y_test)}")


def compare_linear_knn(train_data, test_data, full_data=False, linear_penalty='l2'):
    X_train, y_train = train_data[:, 1:], train_data[:, 0]
    X_test, y_test = test_data[:, 1:], test_data[:, 0]
    if not full_data:  # pick only 2'sand 3's
        train_mask = (y_train == 2) | (y_train == 3)
        test_mask = (y_test == 2) | (y_test == 3)
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]


    # run linear classification with l2 regularisation and log likelihood maximisation
    fit_linear_model(X_train, y_train, X_test, y_test, penalty=linear_penalty, scale=True)

    # run knn classification
    for k in (1,3, 5, 7, 15):
        fit_knn(X_train, y_train, X_test, y_test, k)


if __name__ == '__main__':
    train_data = np.loadtxt(train_data)
    test_data = np.loadtxt(test_data)

    full_data = False
    compare_linear_knn(train_data, test_data, full_data, linear_penalty='l2')
