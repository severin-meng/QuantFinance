# Base Libraries
import os

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    classification_report,
)

from build_feature_data import get_feature_data
from fit_svc import remove_features

# Ignore warnings - optional
import warnings
# Removes warnings in the current job
warnings.filterwarnings("ignore")
# Removes warnings in the spawned jobs
os.environ['PYTHONWARNINGS'] = 'ignore'

sns.set(rc={'figure.figsize': (14, 7)})


def display_roc_curve(classifier, X_test, y_test, title_addon='(test set)'):
    disp_roc = RocCurveDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        name='Area under ROC curve')
    disp_roc.ax_.set_title(f'Receiver Operator Characteristic Curve {title_addon}')
    plt.plot([0, 1], [0, 1])  # random -> is below my answer -> we are slightly better than random.
    plt.show()


def display_confusion_matrix(classifier, X_test, y_test, title_addon='(test set)'):
    disp = ConfusionMatrixDisplay.from_estimator(
        classifier,
        X_test,
        y_test,
        display_labels=classifier.classes_,
        cmap=plt.cm.Blues
    )
    disp.ax_.set_title(f'Confusion matrix {title_addon}')
    plt.show()


def model_score(model, X_train, y_train, X_test, y_test, score_training=True):
    y_pred = model.predict(X_test)

    if score_training:
        y_train_pred = model.predict(X_train)
        print("Classification report on training data")
        print(classification_report(y_train, y_train_pred))

        display_roc_curve(model, X_train, y_train, title_addon='(training set)')
        display_confusion_matrix(model, X_train, y_train, title_addon='(training set)')

    print("Classification report on test data")
    print(classification_report(y_test, y_pred))

    display_roc_curve(model, X_test, y_test)
    display_confusion_matrix(model, X_test, y_test)


def main(base=True, tuned=True):
    _, X_train, y_train, X_test, y_test = get_feature_data()

    keep_features = ['H-L', 'RET_3', 'MOM_10', 'MOM_50']  # MI
    X_train, X_test = remove_features(X_train, X_test, keep_features)

    if base:
        base_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(class_weight='balanced', C=1e-3, cache_size=4000))
        ])
        base_model.fit(X_train, y_train)

        print("Base score: ")
        model_score(base_model, X_train, y_train, X_test, y_test, score_training=True)

    if tuned:
        tuned_model = Pipeline([
            ('scaler', RobustScaler()),
            ('classifier',
             SVC(C=2.848035868435805e-06, cache_size=4000, class_weight={1: 1.5778640776699029, 0: 0.7387944358578052},
                 coef0=1, degree=9, gamma='auto', kernel='poly', max_iter=2000000))
        ])

        tuned_model.fit(X_train, y_train)

        print("Tuned score: ")
        model_score(tuned_model, X_train, y_train, X_test, y_test, score_training=True)


if __name__ == '__main__':
    main(base=False, tuned=True)
