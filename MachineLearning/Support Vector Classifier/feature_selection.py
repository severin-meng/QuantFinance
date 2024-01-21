import os

import numpy as np
import seaborn as sns
import shap

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC


from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    RFECV,
    SequentialFeatureSelector,
    mutual_info_classif,
    f_classif,
    SelectFpr,
    SelectFdr,
    SelectFwe)


from sklearn.model_selection import TimeSeriesSplit

import time

from build_feature_data import get_feature_data, get_class_balance

import winsound

# Ignore warnings - optional
import warnings
# Removes warnings in the current job
warnings.filterwarnings("ignore")
# Removes warnings in the spawned jobs
os.environ['PYTHONWARNINGS'] = 'ignore'

sns.set(rc={'figure.figsize': (14, 7)})


def correlation_test(x_data, rounding=2, drop=False, filter_threshold=0.95):
    sns.set(rc={'figure.figsize': (20, 10)})

    corr_data = x_data.corr()
    sns.heatmap(corr_data,
                annot=True,
                annot_kws={"size": 11},
                fmt=f".{rounding}f",
                linewidth=.5,
                cmap="coolwarm",
                cbar=False)
    plt.title('Features Set Correlations')
    plt.show()
    if drop:
        return filter_correlations(corr_data, threshold=filter_threshold)


def filter_correlations(correlation_data, threshold=0.95):
    abs_corr = correlation_data.abs()
    all_features = correlation_data.columns.values
    remove_features = {'Open', 'High', 'Low', 'Close'}

    for feature in all_features:
        if feature in remove_features:
            continue
        temp_set = {feature}
        correlations = abs_corr[feature].drop(temp_set.union(remove_features), axis=0)
        redundant_features = set(correlations[correlations >= threshold].axes[0])
        remove_features = remove_features.union(redundant_features)
    return list(remove_features)


def run_correlation_tests(train_data, filter_threshold=0.95):
    drop_cols = correlation_test(train_data, rounding=2, filter_threshold=filter_threshold, drop=True)
    print(f"dropping correlated features: {[c for c in train_data.columns.values if c in drop_cols]}")
    X_new = train_data.drop(drop_cols + ['Label'], axis=1)
    correlation_test(X_new, rounding=2)
    return drop_cols


def select_k_best_anova(x, y):
    # taken from sklearn documentation
    selector = SelectKBest(f_classif)
    selector.fit(x, y)
    scores = -np.log10(selector.pvalues_)
    scores /= scores.max()

    x_indices = np.arange(x.shape[-1])
    fig, ax = plt.subplots()
    ax.bar(x_indices - 0.05, scores, width=0.2)
    ax.set_title("ANOVA feature score")
    ax.set_xlabel("Feature number")
    ax.set_ylabel(r"Univariate score")

    rects = ax.patches
    labels = x.columns.values

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 0.02, label, ha="center", va="bottom"
        )

    plt.show()


def mutual_info_base(x, y):
    discrete_mask = [value in ('SIGN',) for value in x.columns.values]  # SIGN is the only discrete feature
    scores = mutual_info_classif(x, y, discrete_features=discrete_mask)
    scores /= scores.max()
    x_indices = np.arange(x.shape[-1])
    fig, ax = plt.subplots()
    ax.bar(x_indices - 0.05, scores, width=0.2)
    ax.set_title("Mutual information feature score")
    ax.set_xlabel("Feature number")
    ax.set_ylabel(r"Univariate score")

    rects = ax.patches
    labels = x.columns.values

    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 0.02, label, ha="center", va="bottom"
        )

    plt.show()


def error_rate_tests(x, y):
    # False positive rate test
    x_fpr = SelectFpr(f_classif, alpha=0.01).fit(x, y)
    print("False positive rate test survivors:")
    # noinspection PyUnresolvedReferences
    print(x_fpr.get_feature_names_out())

    # False discovery rate test
    x_fdr = SelectFdr(f_classif).fit(x, y)
    print("False discovery rate test survivors:")
    # noinspection PyUnresolvedReferences
    print(x_fdr.get_feature_names_out())

    # Family-wise error rate test
    x_fwe = SelectFwe(f_classif).fit(x, y)
    print("Family-wise error test survivors:")
    # noinspection PyUnresolvedReferences
    print(x_fwe.get_feature_names_out())


def recursive_feature_elimination(x, y, weights, n_features=None):
    print("Starting recursive feature elimination...")
    start = time.time()
    estimator = SVC(class_weight=weights, kernel='linear', cache_size=4000)

    selector = RFE(estimator, n_features_to_select=n_features)
    selector = selector.fit(x, y)
    print("Recursive feature finished.")
    print(f"elapsed time: {time.time() - start}")
    # noinspection PyUnresolvedReferences
    print(selector.get_feature_names_out())


def recursive_feature_elimination_cv(x, y, weights, min_features=1):
    print("Starting recursive feature elimination with cross validation...")
    start = time.time()
    tscv = TimeSeriesSplit(n_splits=5, gap=1)

    estimator = SVC(class_weight=weights, kernel='linear', cache_size=4000)
    selector = RFECV(estimator, min_features_to_select=min_features, cv=tscv, n_jobs=-1)
    selector = selector.fit(x, y)
    print("Recursive feature elimination with cross validation finished.")
    print(f"elapsed time: {time.time() - start}")
    # noinspection PyUnresolvedReferences
    print(selector.get_feature_names_out())


def sequential_feature_selection(x, y, weights, n_features='auto', direction='forward'):
    print(f"Starting sequential {direction} feature selection...")
    start = time.time()
    tscv = TimeSeriesSplit(n_splits=5, gap=1)

    estimator = SVC(class_weight=weights, kernel='rbf', cache_size=4000)
    selector = SequentialFeatureSelector(estimator, n_features_to_select=n_features, cv=tscv, direction=direction,
                                         n_jobs=-1)
    selector = selector.fit(x, y)
    print(f"Sequential {direction} feature selection finished.")
    print(f"elapsed time: {time.time() - start}")
    # noinspection PyUnresolvedReferences
    print(selector.get_feature_names_out())


def shap_feature_selection(x, y, weights):
    print("Starting shap feature selection...")
    start = time.time()
    model = Pipeline([
        ('scaler', MinMaxScaler()),
        ('classifier', SVC(class_weight=weights, cache_size=4000))
    ])
    model.fit(x, y)

    x_100 = shap.utils.sample(x, 100, random_state=42)  # check on 100 predictions
    explainer = shap.Explainer(model.predict, x_100, feature_names=x.columns.values)
    shap_values = explainer(x)
    print("Shap feature selection finished.")
    print(f"elapsed time: {time.time() - start}")
    shap.plots.beeswarm(shap_values, max_display=20)
    shap.plots.bar(shap_values.abs.mean(0), max_display=20)


def linear_svm_embedded(X_train, y_train, X_test, weights):
    def fit_predict(estimator, transformer):
        try:
            model = Pipeline([
                ('scaler', transformer),
                ('classifier', estimator)
            ])
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_t_pred = model.predict(X_train)
        except Exception as e:
            print(str(e))
            raise
        return model, model['classifier'].coef_, y_pred, y_t_pred

    def plot_coeff(modelname='L1'):
        coef = []

        if modelname == 'L1':
            c_range = np.logspace(-2, 0.0, 200)
            for i in c_range:
                coef.append(
                    fit_predict(LinearSVC(C=i, penalty='l1', dual=False, max_iter=1000000, class_weight=weights),
                                MinMaxScaler())[1][0])

        elif modelname == 'L2':
            c_range = np.logspace(-3, 0.5, 200)
            for i in c_range:
                coef.append(
                    fit_predict(LinearSVC(C=i, penalty='l2', dual=False, max_iter=1000000, class_weight=weights),
                                MinMaxScaler())[1][0])
        else:
            return

        coefs = np.array(coef)
        num_plots = len(X_train.columns)

        # Plot Coefficients
        plt.figure(figsize=(20, 8))
        plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))
        feature_names = X_train.columns
        for i in range(coefs.shape[-1]):
            plt.plot(c_range, coefs[:, i])
            plt.text(c_range[125], coefs[125, i], f'{feature_names[i]}', size='small')
            plt.text(c_range[-1], coefs[-1, i], f'{feature_names[i]}', size='small')
        plt.xscale('log')
        plt.legend(feature_names, loc='center left')
        plt.title(f'{modelname} coefficients as a function of the regularization')
        plt.xlabel(r'$\lambda$')
        plt.ylabel(r'$\mathbf{w}$')
        return plt.show()

    return plot_coeff


def run_filter_methods(X_train, y_train):
    select_k_best_anova(X_train, y_train)
    mutual_info_base(X_train, y_train)
    error_rate_tests(X_train, y_train)


def run_wrapper_methods(X_train, y_train, balanced_weights):
    # RFE
    recursive_feature_elimination(X_train, y_train, balanced_weights, n_features=None)
    # RFECV
    recursive_feature_elimination_cv(X_train, y_train, balanced_weights, min_features=4)  # takes 4 minutes with n=6

    # sequential selection
    sequential_feature_selection(X_train, y_train, balanced_weights, direction='forward', n_features=5)
    sequential_feature_selection(X_train, y_train, balanced_weights, direction='backward', n_features=5)

    shap_feature_selection(X_train, y_train, balanced_weights)


def run_embedded_methods(X_train, y_train, X_test, balanced_weights):
    embedded_svm = linear_svm_embedded(X_train, y_train, X_test, balanced_weights)
    embedded_svm('L1')
    embedded_svm('L2')


def select_features(run_filter=False, run_wrapper=False, run_embedded=False, drop_correlated_cols=True):
    train_data, X_train, y_train, X_test, y_test = get_feature_data()
    balanced_weights = get_class_balance(y_train)

    if drop_correlated_cols:
        # remove highly correlated features
        correlated_columns = run_correlation_tests(train_data, filter_threshold=0.99)
        X_train = X_train.drop(correlated_columns, axis=1)
        X_test = X_test.drop(correlated_columns, axis=1)

    # Filter methods
    if run_filter:
        run_filter_methods(X_train, y_train)
    # Wrapper methods
    if run_wrapper:
        run_wrapper_methods(X_train, y_train, balanced_weights)
    # Embedded methods
    if run_embedded:
        run_embedded_methods(X_train, y_train, X_test, balanced_weights)


if __name__ == '__main__':
    select_features(
        run_filter=True,
        run_wrapper=True,
        run_embedded=True)
    winsound.Beep(1000, 1000)
