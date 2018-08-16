import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import autosklearn.regression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import mean_squared_error

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
    
    
    
    # Linear Regression
    print("Mean squared error for linear regression: ", round(mean_squared_error(y_test, reg_y_pred)*100))
    print("Accuracy for linear regression:", round(metrics.accuracy_score(y_test, reg_y_pred_int)*100))
      
    
    # Plot results
    # Linear Regression    
    