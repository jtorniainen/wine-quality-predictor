#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 13:27:33 2018

@author: arun
"""

if __name__ == '__main__':
    # Load data from CSV
    # df = pd.read_csv('https://raw.githubusercontent.com/jtorniainen/wine-quality-predictor/master/winequality-white.csv', sep=';')
    df = pd.read_csv('winequality-white.csv', sep=';')

    # Train-Test split
    df_train, df_test = train_test_split(df, test_size=.20, random_state=1)

    X_train_unscaled = df_train.drop('quality', axis=1) 
    y_train = df_train['quality']

    X_test_unscaled = df_test.drop('quality', axis=1)
    y_test = df_test['quality']
    

    # With feature selection
    
    # using extra trees classified
#    from sklearn.ensemble import ExtraTreesClassifier
#    from sklearn.feature_selection import SelectFromModel
#    clf = ExtraTreesClassifier()
#    clf = clf.fit(X_train, y_train)    
#    model = SelectFromModel(clf, prefit=True)
#    X_train_fsel = model.transform(X_train)
#    X_test_fsel = model.transform(X_test)
#    
#    from sklearn.svm import SVC    
#    from sklearn.feature_selection import RFE
#    svc = SVC(kernel="linear", C=1)
#    rfe = RFE(estimator=svc, n_features_to_select=4, step=1)
#    rfe.fit(X_train, y_train)
    
    # using recursive feature elimination
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVR
        
    # code to find out optimal number of features
    for nfeatures in range(1,12):    
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, nfeatures, step=1)
        selector = selector.fit(X_train, y_train)
        print(selector.support_)
        
        features_selected = []
        for i in range(0,selector.support_.size):
            if selector.support_[i]==True:
                features_selected.append(i)
            
        X_train_fsel = X_train[X_train.columns[features_selected]]
        X_test_fsel = X_test[X_test.columns[features_selected]]    
        
        reg = linear_model.LinearRegression()
        reg.fit(X_train_fsel, y_train)
        linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
       
        reg_fsel_y_pred = reg.predict(X_test_fsel)
        reg_fsel_y_pred_int = np.round(reg_fsel_y_pred)
        print("No of features, Mean squared error, Accuracy", nfeatures, round(mean_squared_error(y_test, reg_fsel_y_pred)*100), round(metrics.accuracy_score(y_test, reg_fsel_y_pred_int)*100))
 
#    # optimal number of features was found to be 6.
#    estimator = SVR(kernel="linear")
#    selector = RFE(estimator, 6, step=1)
#    selector = selector.fit(X_train, y_train)
#    print(selector.support_)
#    
#    # select the features, i.e., column numbers
#    features_selected = []
#    for i in range(0,selector.support_.size):
#        if selector.support_[i]==True:
#            features_selected.append(i)
#    
#    X_train_fsel = X_train[X_train.columns[features_selected]]
#    X_test_fsel = X_test[X_test.columns[features_selected]]    
#    
#    reg = linear_model.LinearRegression()
#    reg.fit(X_train_fsel, y_train)
#    linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
#   
#    reg_fsel_y_pred = reg.predict(X_test_fsel)
#    reg_fsel_y_pred_int = np.round(reg_fsel_y_pred)          
#    
#    # Linear Regression with feature selection        
#    print("Mean squared error for linear regression with feature selection: ", round(mean_squared_error(y_test, reg_fsel_y_pred)*100))
#    print("Accuracy for linear regression  with feature selection:", round(metrics.accuracy_score(y_test, reg_fsel_y_pred_int)*100))  