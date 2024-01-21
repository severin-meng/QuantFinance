# Base Libraries
import numpy as np
import os

import seaborn as sns
sns.set(rc={'figure.figsize': (20, 10)})

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC, LinearSVC, NuSVC


from sklearn.model_selection import (
    TimeSeriesSplit,
    )
from sklearn.model_selection import GridSearchCV

# Ignore warnings - optional
import warnings
# Removes warnings in the current job
warnings.filterwarnings("ignore")
# Removes warnings in the spawned jobs
os.environ['PYTHONWARNINGS'] = 'ignore'

from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    mean_squared_error, # do i really need this? for classification
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    roc_auc_score,
    auc,
    make_scorer,
    classification_report,
    confusion_matrix
)

import time

from build_feature_data import get_class_balance, get_feature_data

import winsound

frequency = 1000  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second


def base_svc_model(x, y, weights='balanced'):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(class_weight=weights, cache_size=4000))
    ])
    model.fit(x, y)
    return model


def base_linear_model(x, y, weights='balanced'):
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LinearSVC(class_weight=weights, penalty='l1', dual=False))
    ])
    model.fit(x, y)
    return model


def base_nusvc_model(x, y, weights='balanced'):
    nu_max = upper_nu_bound(y)
    print(f"upper bound on nu: {nu_max}")
    model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', NuSVC(nu=nu_max*0.0202, class_weight='balanced'))
    ])
    model.fit(x, y)
    return model


def precision_1_score(y_true, y_pred):
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

    return precision_1**4 * precision_0 * recall_1**2 * recall_0**4


def display_roc_curve(classifier, X_test, y_test):
    disp_roc = RocCurveDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        name='placeholder')
    disp_roc.ax_.set_title('ROC Curve')
    plt.plot([0, 1], [0, 1])  # random -> is below my answer -> we are slightly better than random.
    plt.show()


def display_confusion_matrix(classifier, X_test, y_test):
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=classifier.classes_,
        cmap=plt.cm.Blues
    )
    disp.ax_.set_title('Confusion matrix')
    plt.show()


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

    # display_roc_curve(model, X_test, y_test)
    # display_confusion_matrix(model, X_test, y_test)
    return "\n" + report1 +"\n" + report2


def tune_hyperparams_svc(X_train, y_train, weights='balanced', score_func='roc_auc'):
    model = base_svc_model(X_train, y_train)
    bias_range = [0.85, 0.9, 0.95, 0.98, 1, 1.02, 1.05, 1.1, 1.15]

    prefix = 'classifier__'
    param_grid = {
        'scaler': [RobustScaler(), StandardScaler(), MinMaxScaler()],
        f'{prefix}C': list(np.logspace(-8, 0, 30)),
        f'{prefix}class_weight': [get_class_balance(y_train, bias) for bias in bias_range],
        f'{prefix}cache_size': [4000],
        f'{prefix}degree': np.arange(2, 10, 1),
        f'{prefix}gamma': ['scale', 'auto'],
        f'{prefix}kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        f'{prefix}max_iter': [10000],
        f'{prefix}coef0': [0, 1, 2]
    }

    return tune_hyperparams(model, param_grid, X_train, y_train, scoring=score_func)


def tune_hyperparams_l1_svc(X_train, y_train):
    model = base_linear_model(X_train, y_train)

    prefix = 'classifier__'
    param_grid = {
        'scaler': [StandardScaler(), MinMaxScaler(), RobustScaler()],
        f'{prefix}C': list(np.logspace(-3, 0.5, 100)),
        # f'{prefix}class_weight': ['balanced', get_class_balance(y_train, bias=0.9), get_class_balance(y_train, bias=1.05),
        #                           get_class_balance(y_train, bias=1.1)],
        f'{prefix}class_weight': [get_class_balance(y_train, bias=10)],
        f'{prefix}dual': [False],
        f'{prefix}fit_intercept': [True, False],
        f'{prefix}loss': ['squared_hinge'],
        f'{prefix}max_iter': [100000],
        f'{prefix}penalty': ['l1'],
    }
    return tune_hyperparams(model, param_grid, X_train, y_train)


def tune_hyperparams_nu_svc(X_train, y_train):
    model = base_nusvc_model(X_train, y_train)

    nu_max = upper_nu_bound(y_train)
    print(f"upper bound on nu: {nu_max}")

    prefix = 'classifier__'
    param_grid = {
        'scaler': [MinMaxScaler()],  # [StandardScaler(), MinMaxScaler(), RobustScaler()],
        f'{prefix}nu': list(np.linspace(0.001, nu_max-0.01, 100)),
        # f'{prefix}class_weight': ['balanced', get_class_balance(y_train, bias=0.9), get_class_balance(y_train, bias=1.05),
        #                           get_class_balance(y_train, bias=1.1)],
        f'{prefix}class_weight': [get_class_balance(y_train, bias=3)],
        f'{prefix}cache_size': [4000],
        f'{prefix}degree': [2],  # np.arange(2, 10, 1),
        f'{prefix}gamma': ['scale', 'auto'],
        f'{prefix}kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        f'{prefix}max_iter': [10000],
        f'{prefix}coef0': [0, 1, 2]
    }
    return tune_hyperparams(model, param_grid, X_train, y_train)


def tune_hyperparams(model, param_grid, X_train, y_train, scoring='roc_auc'):
    start = time.time()
    tscv = TimeSeriesSplit(n_splits=5, gap=1)

    grid_search = GridSearchCV(model, param_grid, scoring=scoring, n_jobs=-1, cv=tscv, verbose=1)
    grid_search.fit(X_train, y_train)
    end = time.time()
    print(f"Elapsed time: {end - start}")

    # Best Params & Scores
    print(f"Optimal Parmas: {grid_search.best_params_}, Best Scores: {round(grid_search.best_score_, 4)}")

    text = f"Optimal Parmas: {grid_search.best_params_}, Best Scores: {round(grid_search.best_score_, 4)}"

    # tune model to optimal params
    model.set_params(**grid_search.best_params_)
    text += f"\nfitting model with optimal params: {model.get_params()}\n"
    print(f"fitting model with optimal params: {model.get_params()}")
    model.fit(X_train, y_train)
    return model, text


def remove_features(X_train, X_test, keep_features=None):
    # features_to_use = ['Open', 'Low', 'Close', 'O-C', 'H-L', 'RET_1', 'RET_2', 'RET_3', 'MOM_5', 'MOM_10'] for standardized threshold -1.167
    features_to_use = ['H-L', 'SIGN', 'MOM_5', 'MOM_20', 'MOM_30', 'MOM_50']  # 6 top mixed features
    # features_to_use = ['H-L', 'SIGN', 'MOM_5', 'MOM_20', 'MOM_30', 'MOM_50', 'O-C', 'RET_1', 'RET_2', 'RET_3']  # 10 top features

    #  features_to_use = ['H-L', 'SIGN', 'RET_2', 'RET_3', 'RET_4', 'RET_5']  # 6 SVC-top features (wrapper only)
    # features_to_use = ['H-L', 'SIGN', 'RET_2', 'RET_3', 'RET_4', 'RET_5', 'Open', 'RET_1', 'MOM_30']  # 9 SVC-top features
    if keep_features is not None:
        all_features = X_train.columns.values
        drop_features = [f for f in all_features if f not in keep_features]
        X_train = X_train.drop(drop_features, axis=1)
        X_test = X_test.drop(drop_features, axis=1)
    else:
        features_to_use = ['MA_5', 'H-L', 'SIGN', 'MOM_5', 'MOM_20',  'RET_2', 'RET_3', 'MOM_10', 'MOM_2']  # 10 SVC-top features (wrapper only)

        features_to_drop = [f for f in X_train.columns.values if f not in features_to_use]
        X_train = X_train.drop(features_to_drop, axis=1)
        X_test = X_test.drop(features_to_drop, axis=1)

    return X_train, X_test


def write_fit_results(bias, features, params, score, counter=1):
    path = r'D:\CQF\Module 4\Assignment\code_neu_0.3262 - gap1\fit_mi_rocauc'
    filename = os.sep.join([path, f'res_{counter}.txt'])
    with open(filename, 'w') as file:
        file.write(f"score: {score}\n")
        file.write(f"params: {params}\n")
        file.write(f"features: {features}\n")
        file.write(f"\nBias is: {bias}")


def upper_nu_bound(y_train):
    m = y_train.shape[0]
    mp = np.sum(y_train)
    mm = m - mp
    return 2*min(mp, mm) / m


if __name__ == '__main__':
    _, X_train, y_train, X_test, y_test = get_feature_data()
    xtr = X_train.copy()
    xte = X_test.copy()
    # X_train, X_test = remove_features(X_train, X_test)
    balanced_weights = get_class_balance(y_train)

    imbalanced_weights = get_class_balance(y_train, bias=0.98) #  1.0000001 goes to 1, 1 goest to 1 also, 0.9 goes to 0
    # 0.98 works for both, 0.96 too for 10 mixedfeatures
    # 1 goes to 1
    # 0.94 goes to 0

    # 0.999999 goes to 0 for 6 mixed features
    # 1.0 goes to 1 for 6 mixed features

    # 0.98 for 6-SVC features goes to 0
    # 1.02 for 6-SVC features goes to 1

    # 10 svc feautres: 0.98 goes to 0, 1.02 goes to 1


    # imbalanced_weights = get_class_balance(y_train, bias=1.5)

    bias_list = [0.85, 0.9, 0.95, 0.98, 1, 1.02, 1.05, 1.1, 1.15]
    # bias_list = [0.99, 1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09]

    # ['O-C', 'H-L', 'RET_1', 'RET_2', 'RET_3', 'RET_4', 'RET_5', 'SIGN', 'MOM_2', 'MOM_5', 'MOM_10', 'MOM_20', 'MOM_30', 'MOM_50', 'MA_5', 'MA_20', 'MA_50']  # full

    # features list for gap = 51
    features_list = [
        ['O-C', 'H-L', 'RET_1', 'RET_2', 'RET_3', 'RET_4', 'RET_5', 'SIGN', 'MOM_2', 'MOM_5', 'MOM_10', 'MOM_20',
         'MOM_30', 'MOM_50', 'MA_5', 'MA_20', 'MA_50'],
        ['H-L', 'MOM_2', 'MOM_5', 'MOM_20', 'MOM_30', 'MOM_50'],  # FPR
        ['O-C', 'H-L', 'RET_1', 'MOM_2', 'MOM_5', 'MOM_10', 'MOM_20', 'MOM_30', 'MOM_50'],  # FDR
        ['H-L', 'MOM_5', 'MOM_20', 'MOM_30', 'MOM_50'],  # FWR
        ['O-C', 'RET_3', 'RET_5', 'SIGN', 'MOM_2', 'MA_5'],  # RFE
        ['RET_3', 'RET_5', 'SIGN', 'MOM_2'],  # RFECV
        ['O-C', 'RET_1', 'RET_3', 'RET_5', 'SIGN', 'MOM_2', 'MOM_20', 'MA_5', 'MA_20'],  # RFECV 2
        ['O-C', 'H-L', 'RET_1', 'RET_2', 'RET_3', 'RET_4', 'MOM_50', 'MA_5'],  # SEQ FWD
        ['O-C', 'H-L', 'RET_1', 'RET_2', 'MOM_50', 'MA_5'],  # SEQ FWD 2
        ['RET_4', 'RET_5', 'SIGN', 'MOM_2', 'MOM_20', 'MOM_30', 'MOM_50', 'MA_5', 'MA_20'],  # SEQ BWD
        ['H-L', 'SIGN', 'MOM_2', 'MOM_5', 'MOM_20', 'MOM_30', 'MOM_50', 'MA_5', 'MA_20', 'MA_50'],  # SHAP 1
        ['SIGN', 'MOM_20', 'MOM_30', 'MOM_50', 'MA_5', 'MA_20', 'MA_50'],  # SHAP 2
        ['H-L', 'RET_3', 'MOM_10', 'MOM_50'],  # MI 1
        ['H-L', 'RET_3', 'MOM_10', 'MOM_50', 'RET_2', 'MOM_20'],  # MI 2
        ['H-L', 'RET_3', 'MOM_10', 'MOM_50', 'RET_2', 'MOM_20', 'O-C', 'RET_5', 'MA_5'],  # MI 3
        ['H-L', 'MOM_5', 'MOM_20', 'MOM_30', 'MOM_50'],  # Anova 1
        ['H-L', 'MOM_5', 'MOM_20', 'MOM_30', 'MOM_50', 'O-C', 'RET_1', 'MOM_2', 'MOM_10'],  # Anova 2
    ]

    # features list for gap = 1 and precision optimum
    features_list = [
        ['H-L', 'RET_3', 'MOM_10', 'MOM_50'],  # MI 1, same
        ['H-L', 'MOM_2', 'MOM_5', 'MOM_20', 'MOM_30', 'MOM_50'],  # FPR, same
        ['O-C', 'H-L', 'RET_1', 'MOM_2', 'MOM_5', 'MOM_10', 'MOM_20', 'MOM_30', 'MOM_50'],  # FDR, same
        ['H-L', 'MOM_5', 'MOM_20', 'MOM_30', 'MOM_50'],  # FWR, same
        ['O-C', 'RET_1', 'SIGN', 'MOM_2', 'MOM_30', 'MA_5', 'MA_20', 'MA_50'],  # RFE, new
        ['O-C', 'SIGN', 'MOM_2', 'MA_5', 'MA_50'],  # RFECV, new, fixed
        ['H-L', 'RET_3', 'MOM_10', 'MOM_50', 'RET_2', 'MOM_20'],  # MI 2, same
        ['H-L', 'MOM_5', 'MOM_20', 'MOM_30', 'MOM_50'],  # Anova 1, same
        ['H-L', 'MOM_5', 'MOM_20', 'MOM_30', 'MOM_50', 'O-C', 'MOM_2', 'MOM_10'],  # Anova 2, new
        ['H-L', 'MOM_5', 'MOM_20', 'MOM_30', 'MOM_50', 'RET_1', 'MOM_2', 'MOM_10'],  # Anova 3
    ]


    """
    # mutual info features
    features_list = [
        ['H-L', 'RET_3', 'MOM_10', 'MOM_50'],  # MI 1, same
        ['H-L', 'MOM_10', 'MOM_50'],
        ['RET_3', 'MOM_10', 'MOM_50'],
        ['H-L', 'RET_3', 'MOM_50'],
        ['H-L', 'RET_3', 'MOM_10'],
        ]

    # precision optimized features
    features_list = [
        ['O-C', 'RET_1', 'RET_2', 'RET_3', 'SIGN', 'MOM_2', 'MOM_5', 'MOM_10', 'MOM_20',
         'MOM_30', 'MA_5', 'MA_20', 'MA_50'],  # rfecv 2
        ['H-L', 'RET_1', 'RET_2', 'RET_3', 'MOM_5'],
        ['MOM_5', 'MOM_50', 'MA_5', 'MA_20', 'MA_50'],
    ]
    """
    custom_scoring = make_scorer(precision_1_score, greater_is_better=True)

    features = str(X_train.columns.values)

    nu_svc = False
    svc = True
    linear = False
    if nu_svc:
        # seems to ignore class balance  -> do not use this model
        # nusvc
        base_model = base_nusvc_model(X_train, y_train, balanced_weights)
        # model = tune_hyperparams_nu_svc(X_train, y_train)

        # winsound.Beep(frequency, duration)

        # model_score(model, X_train, y_train, X_test, y_test)
        model_score(base_model, X_train, y_train, X_test, y_test)

    if svc:
        features_list = [
            ['O-C', 'H-L', 'SIGN', 'MOM_30', 'MOM_50', 'MA_5']
        ]
        """
        X_train, X_test = remove_features(xtr, xte, keep_features=features_list[0])
        model = Pipeline([
            ('scaler', MinMaxScaler()),
            ('classifier',
             SVC(class_weight={1: 1.5623948220064725, 0: 0.7387944358578052}, C=0.003290344562312671, cache_size=4000,
                 degree=5, gamma='scale', kernel='poly', coef0=0))
        ])
        model.fit(X_train, y_train)
        model_score(model, X_train, y_train, X_test, y_test)
        display_roc_curve(model, X_test, y_test)
        """
        counter=1
        for feature_lst in features_list:
            X_train, X_test = remove_features(xtr, xte, keep_features=feature_lst)
            try:
                # imbalanced_weights = get_class_balance(y_train, bias=bias)

                # this actually depends on the class balance
                # svc
                # base_model = base_svc_model(X_train, y_train, imbalanced_weights)
                model, params = tune_hyperparams_svc(X_train, y_train, weights=imbalanced_weights,
                                                     score_func='roc_auc')

                score = model_score(model, X_train, y_train, X_test, y_test)
                # model_score(base_model, X_train, y_train, X_test, y_test)

                write_fit_results('Auto', feature_lst, params, score, counter=counter)
            except Exception:
                write_fit_results('Auto', feature_lst, "ERROR", "ERROR", counter=counter)
                counter += 1
                continue
            counter += 1

        winsound.Beep(frequency, duration)

    if linear:
        # this actually depends on the class balance
        # linear svc l1
        base_model = base_linear_model(X_train, y_train, imbalanced_weights)
        model = tune_hyperparams_l1_svc(X_train, y_train)

        winsound.Beep(frequency, duration)

        model_score(model, X_train, y_train, X_test, y_test)
        model_score(base_model, X_train, y_train, X_test, y_test)

    print("*****DONE*****")
