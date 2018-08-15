import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import autosklearn.regression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
import uuid

if __name__ == '__main__':
    task_time = 3600
    per_run_time = 360
    # Load data from CSV
    # df = pd.read_csv('https://raw.githubusercontent.com/jtorniainen/wine-quality-predictor/master/winequality-white.csv', sep=';')
    df = pd.read_csv('winequality-white.csv', sep=';')

    # Train-Test split
    df_train, df_test = train_test_split(df, test_size=.20, random_state=1)

    X_train = df_train.drop('quality', axis=1)
    y_train = df_train['quality']

    X_test = df_test.drop('quality', axis=1)
    y_test = df_test['quality']

    # ----- 2. Automatic -----

    # 2a. AutoML with preprocessing
    feature_types = ['numerical'] * 11
    tmp_folder = str(uuid.uuid4())
    output_folder = str(uuid.uuid4())

    automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=task_time,
                                                         per_run_time_limit=per_run_time,
                                                         tmp_folder=tmp_folder,
                                                         output_folder=output_folder)

    # Fit data
    automl.fit(X_train, y_train, dataset_name='wine-quality', feat_type=feature_types)

    # Get prediction
    y_pred = automl.predict(X_test)
    y_pred_int = np.round(y_pred)
    r2 = r2_score(y_test, y_pred_int)
    mse = mean_squared_error(y_test, y_pred_int)

    # 2b. AutoML without preprocessing
    tmp_folder = str(uuid.uuid4())
    output_folder = str(uuid.uuid4())
    automl_no_pp = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=task_time,
                                                         per_run_time_limit=per_run_time,
                                                         tmp_folder=tmp_folder,
                                                         output_folder=output_folder,
                                                         include_preprocessors=['no_preprocessing'])

    # Fit data
    automl_no_pp.fit(X_train, y_train, dataset_name='wine-quality', feat_type=feature_types)

    # Get prediction for test set
    y_pred_no_pp = automl_no_pp.predict(X_test)
    y_pred_no_pp_int = np.round(y_pred)
    r2_no_pp = r2_score(y_test, y_pred_no_pp_int)
    mse_no_pp = mean_squared_error(y_test, y_pred_no_pp_int)
