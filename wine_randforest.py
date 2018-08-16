#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:11:32 2018

@author: UWT
"""

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
    
    from sklearn.ensemble import RandomForestClassifier    
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    rfc_y_pred = clf.predict(X_test)
    rfc_y_pred_int = np.round(rfc_y_pred)    
    
    print("Mean squared error for random forest: ", round(mean_squared_error(y_test, rfc_y_pred_int)*100))
    print("Accuracy for random forest:", round(metrics.accuracy_score(y_test, rfc_y_pred_int)*100))
    
    