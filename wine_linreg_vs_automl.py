import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import autosklearn.regression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == '__main__':
    # Load data from CSV
    # df = pd.read_csv('https://raw.githubusercontent.com/jtorniainen/wine-quality-predictor/master/winequality-white.csv', sep=';')
    df = pd.read_csv('winequality-white.csv', sep=';')

    # Train-Test split
    df_train, df_test = train_test_split(df, test_size=.20, random_state=1)

    X_train = df_train.drop('quality', axis=1)
    y_train = df_train['quality']

    X_test = df_test.drop('quality', axis=1)
    y_test = df_test['quality']

    # ----- 1. Manual -----
    # ----- 1a. Linear regression -----

    reg = linear_model.LinearRegression()
    reg.fit(X_train, y_train)
    linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

    #print(reg.coef_)
    reg_y_pred = reg.predict(X_test)
    reg_y_pred_int = np.round(reg_y_pred)
    
    f, ax = plt.subplots(1)
    sns.distplot(y_test, kde=False, ax=ax)
    sns.distplot(reg_y_pred_int, kde=False, ax=ax)
    ax.set_title('Linear Regression')
    
    # ----- 1b. Random Forest -----
    
    from sklearn.ensemble import RandomForestClassifier    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    rfc_y_pred = clf.predict(X_test)
    rfc_y_pred_int = np.round(rfc_y_pred)
    
    # ----- 2. Automatic -----
    # Load data from CSV


    feature_types = ['numerical'] * 11

    # Initialize auto learning object
    automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=120,
                                                         per_run_time_limit=30,
                                                         tmp_folder='/tmp/kekkonen1412412412413412124',
                                                         output_folder='/tmp/wine-out1241243124kj214124412124kjj')

    # Fit data
    automl.fit(X_train, y_train, dataset_name='wine-quality', feat_type=feature_types)

    # Get prediction for test set
    y_pred = automl.predict(X_test)
    y_pred_int = np.round(y_pred)
    r2 = r2_score(y_test, y_pred_int)
    mse = mean_squared_error(y_test, y_pred_int)
    #print("AutoMLAccuracy:", metrics.accuracy_score(y_test, y_pred))
    # Visualize results
    f, ax = plt.subplots(1)
    sns.distplot(y_test, kde=False, ax=ax)
    sns.distplot(y_pred_int, kde=False, ax=ax)
    ax.set_title('AutoML')

    plt.show()

    # 3. Comparison of results from 1 and 2
    
    # Linear Regression
    print("Mean squared error for linear regression: ", mean_squared_error(y_test, reg_y_pred))
    print("Accuracy for linear regression:", metrics.accuracy_score(y_test, reg_y_pred_int))   
    # Random Forest
    print("Mean squared error for random forest: ", mean_squared_error(y_test, rfc_y_pred_int))
    print("Accuracy for random forest:", metrics.accuracy_score(y_test, rfc_y_pred_int))    