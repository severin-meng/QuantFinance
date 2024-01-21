# Base Libraries
import numpy as np
import os

import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import (
    TimeSeriesSplit,
    )
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    make_scorer,
    classification_report,
)
import time

from build_feature_data import get_class_balance, get_feature_data

import winsound

# Ignore warnings - optional
import warnings
# Removes warnings in the current job
warnings.filterwarnings("ignore")
# Removes warnings in the spawned jobs
os.environ['PYTHONWARNINGS'] = 'ignore'

sns.set(rc={'figure.figsize': (20, 10)})


def base_svc_model(x, y, weights='balanced'):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(class_weight=weights, cache_size=4000))
    ])
    model.fit(x, y)
    return model


def precision_1_score(y_true, y_pred):
    # custom scoring function
    true_pred_1 = np.sum(y_true * y_pred)

    # flip labels:
    y_t0 = 1 - y_true; y_p0 = 1 - y_pred
    true_pred_0 = np.sum(y_t0 * y_p0)

    if np.sum(y_pred) == 0 or np.sum(y_true) == 0 or np.sum(y_p0) == 0 or np.sum(y_t0) == 0:
        return 0  # score zero for a model predicting only zeros

    precision_1 = true_pred_1 / np.sum(y_pred)
    recall_1 = true_pred_1 / np.sum(y_true)

    precision_0 = true_pred_0 / np.sum(y_p0)
    recall_0 = true_pred_0 / np.sum(y_t0)

    return precision_1**4 * precision_0 * recall_1**2 * recall_0**2


def model_score(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    print("Classification report on test data")
    report2 = "\nClassification report on test data\n"
    report2 += str(classification_report(y_test, y_pred))
    print(report2)

    print("Classification report on training data")
    report1 = "\nClassification report on training data\n"
    report1 += str(classification_report(y_train, y_train_pred))
    print(report1)
    return "\n" + report1 + "\n" + report2


def tune_hyperparams_svc(X_train, y_train, score_func='roc_auc'):
    model = base_svc_model(X_train, y_train)

    bias_range = [0.85, 0.9, 0.95, 0.98, 1, 1.02, 1.05, 1.1, 1.15]

    prefix = 'classifier__'
    param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
        f'{prefix}C': list(np.logspace(-8, 0, 50)),
        f'{prefix}class_weight': [get_class_balance(y_train, bias) for bias in bias_range],
        f'{prefix}cache_size': [4000],
        f'{prefix}degree': np.arange(2, 10, 1),
        f'{prefix}gamma': ['scale', 'auto'],
        f'{prefix}kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        f'{prefix}max_iter': [10000],
        f'{prefix}coef0': [0, 1, 2]
    }

    return tune_hyperparams(model, param_grid, X_train, y_train, scoring=score_func)


def tune_hyperparams(model, param_grid, X_train, y_train, scoring='roc_auc'):
    start = time.time()
    tscv = TimeSeriesSplit(n_splits=5, gap=1)

    grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=tscv, verbose=1)
    grid_search.fit(X_train, y_train)
    end = time.time()
    print(f"Elapsed time: {end - start}")

    # Best Params & Scores
    print(f"Optimal Parmas: {grid_search.best_params_}, Best Scores: {round(grid_search.best_score_, 4)}")

    # tune model to optimal params
    model.set_params(**grid_search.best_params_)
    model.fit(X_train, y_train)
    return model


def remove_features(X_train, X_test, keep_features):
    all_features = X_train.columns.values
    drop_features = [f for f in all_features if f not in keep_features]
    X_train = X_train.drop(drop_features, axis=1)
    X_test = X_test.drop(drop_features, axis=1)
    return X_train, X_test


if __name__ == '__main__':
    _, X_train, y_train, X_test, y_test = get_feature_data()
    xtr = X_train.copy()
    xte = X_test.copy()

    # custom scoring - replace the score_func argument in the tune_hyperparams_svc function call
    custom_scoring = make_scorer(precision_1_score, greater_is_better=True)

    # adjust this list of features if you wish to optimize over different features
    features = ['H-L', 'SIGN', 'RET_2', 'RET_3', 'RET_4', 'RET_5']

    X_train, X_test = remove_features(xtr, xte, features)
    model = tune_hyperparams_svc(X_train, y_train, score_func='roc_auc')
    score = model_score(model, X_train, y_train, X_test, y_test)
    # optionally write the score to a file to automate the process

    winsound.Beep(1000, 1000)
