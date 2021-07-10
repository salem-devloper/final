# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for statistical data visualization
import argparse
import os
#matplotlib inline

# Any results you write to the current directory are saved as output.
def get_args():

    parser = argparse.ArgumentParser(description = "U-Net for Lung Segmentation" ,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set your environment
    parser.add_argument('--path', type=str, default = 'E:/2 MASTER/Memoire/07-06-2021 (croped)')
    return parser.parse_args()

def main():

#data = '/kaggle/input/predicting-a-pulsar-star/pulsar_stars.csv'

    args = get_args()

    df = pd.read_csv(os.path.join(args.path,'data_concat_non_normaliz.csv'))

    # Declare feature vector and target variable

    X = df.drop(['index', 'img', 'target'], axis=1)

    y = df['target']

    # split X and y into training and testing sets

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


    cols = X_train.columns

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=[cols])

    X_test = pd.DataFrame(X_test, columns=[cols])

    #Run SVM with default hyperparameters 
    #Table of Contents
    #Default hyperparameter means C=1.0, kernel=rbf and gamma=auto among other parameters.

    # import SVC classifier
    from sklearn.svm import SVC


    # import metrics to compute accuracy
    from sklearn.metrics import accuracy_score


    # instantiate classifier with default hyperparameters
    svc=SVC() 


    # fit classifier to training set
    svc.fit(X_train,y_train)


    # make predictions on test set
    y_pred=svc.predict(X_test)


    # compute and print accuracy score
    print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    #Run SVM with rbf kernel and C=100.0
    #We have seen that there are outliers in our dataset. 
    #So, we should increase the value of C as higher C means fewer outliers. 
    #So, I will run SVM with kernel=rbf and C=100.0.

    # instantiate classifier with rbf kernel and C=100
    svc=SVC(C=100.0) 


    # fit classifier to training set
    svc.fit(X_train,y_train)


    # make predictions on test set
    y_pred=svc.predict(X_test)


    # compute and print accuracy score
    print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    #Run SVM with rbf kernel and C=1000.0

    # instantiate classifier with rbf kernel and C=1000
    svc=SVC(C=1000.0) 


    # fit classifier to training set
    svc.fit(X_train,y_train)


    # make predictions on test set
    y_pred=svc.predict(X_test)


    # compute and print accuracy score
    print('Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

    #13. Run SVM with linear kernel 

    #Run SVM with linear kernel and C=1.0
    # instantiate classifier with linear kernel and C=1.0
    linear_svc=SVC(kernel='linear', C=1.0) 


    # fit classifier to training set
    linear_svc.fit(X_train,y_train)


    # make predictions on test set
    y_pred_test=linear_svc.predict(X_test)


    # compute and print accuracy score
    print('Model accuracy score with linear kernel and C=1.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_test)))

    #Compare the train-set and test-set accuracy
    #Now, I will compare the train-set and test-set accuracy to check for overfitting.

    y_pred_train = linear_svc.predict(X_train)

    print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))

    #Check for overfitting and underfitting
    # print the scores on training and test set

    print('Training set score: {:.4f}'.format(linear_svc.score(X_train, y_train)))

    print('Test set score: {:.4f}'.format(linear_svc.score(X_test, y_test)))

if __name__ == '__main__':

    main()