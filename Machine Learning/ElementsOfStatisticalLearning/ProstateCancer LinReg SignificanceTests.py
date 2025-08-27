import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats

base_path = r"C:\Quantitative Finance\Elements of Statistical Learning\prostate dataset"
prostate_data = os.path.join(base_path, "prostate.data.txt")


def fit_linear_model(X_train, y_train, X_test, y_test):
    pipe = Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())])
    pipe.fit(X_train, y_train)
    X_test_scaled = (X_test - X_train.mean()) / (X_train.std()) * np.sqrt(scaled_variance)
    X_train_scaled = (X_train - X_train.mean()) / (X_train.std()) * np.sqrt(scaled_variance)
    lin = LinearRegression()
    # lin.n_iter_ = 500
    lin.fit(X_train_scaled, y_train)
    print(f"Linear training performance: {lin.score(X_train_scaled, y_train)}")
    print(f"Linear test performance: {lin.score(X_test_scaled, y_test)}")


def fit_linear_model_manual(X_train, y_train, X_test, y_test):
    N, p = X_train.shape
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X = np.c_[np.ones(N), X_train_scaled]  # add intercept as extra feature
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y_train

    rss = np.linalg.norm(X @ beta - y_train)**2
    variance_regressor = rss / (N - p - 1)
    std_error = np.sqrt(variance_regressor * np.diagonal(xtx_inv))
    z_scores = beta / std_error

    p_values = 1-stats.t.cdf(z_scores, N-p-1)

    # table 3.2
    table_32 = pd.DataFrame(data={'Coefficient': beta, 'Std. Error': std_error, 'Z Score': z_scores, 'p-Values': p_values},
                            index = ["Intercept"] + X_train.columns.to_list())
    print(table_32)

    drop_predictors = table_32[abs(table_32["Z Score"]) < 2].index
    p_0 = p - drop_predictors.shape[0]
    X_train_0 = X_train.drop(columns=drop_predictors)
    X_train_0_scaled = scaler.fit(X_train_0).transform(X_train_0)
    X_0 = np.c_[np.ones(N), X_train_0_scaled]
    xtx_inv_0 = np.linalg.inv(X_0.T @ X_0)
    beta_0 = xtx_inv_0 @ X_0.T @ y_train
    rss0 = np.linalg.norm(X_0 @ beta_0 - y_train)**2

    f_statistic = (rss0 - rss) / (p - p_0) / (rss / (N - p - 1))
    f_p_value = 1 - stats.f.cdf(f_statistic, p - p_0, N - p - 1)  # prob of f > f_statistic
    print(f"F-Statistic: Pr(F({p-p_0}, {N-p-1}) > {f_statistic:.2f}) = {f_p_value:.2f}")



if __name__ == '__main__':
    full_data = pd.read_csv(prostate_data, delimiter='\t', index_col=0)
    train_data = full_data[full_data['train'] == 'T'].drop(columns=['train'])
    test_data = full_data[full_data['train'] == 'F'].drop(columns=['train'])

    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]

    # training correlation between predictors
    print(X_train.corr())



