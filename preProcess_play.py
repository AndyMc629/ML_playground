#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 12:39:49 2017

@author: apm13

Let's play with some data pre-processing in python.

https://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/

Random: interesting chat on stackoverflow about importing python lib's at top versus
dynamically throughout the script/program. I like doing it in functions in case
user needs/wants a different library.
http://stackoverflow.com/questions/128478
/should-python-import-statements-always-be-at-the-top-of-a-module

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

# personal note: I hate it when a tutorial gives you data pre-packaged like this,
# it's sooo unatural .... >.<
X_train = pd.read_csv("../data/loan_prediction/X_train.csv")
Y_train = pd.read_csv("../data/loan_prediction/Y_train.csv")

X_test = pd.read_csv("../data/loan_prediction/X_test.csv")
Y_test = pd.read_csv("../data/loan_prediction/Y_test.csv")

# let's have a look.
plt.figure(1)
X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])
plt.show()
plt.close()                        
                        
#
# Currently the numerical data is all over the place in terms of normalisation.
# Let's run a kNN clustering algorithm to see how this impacts the prediction.
#
# Note: need to ravel() the y_values to stop a data conversion warning (expects a 1d array).
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                 'Loan_Amount_Term', 'Credit_History']], Y_train.values.ravel())

# check performance of model on test set, remember should only ever do this at end
# in a normal analysis!
print "Accuracy_score of 5NN without normalisation = ", float(accuracy_score(Y_test, knn.predict(X_test[['ApplicantIncome' ,'CoapplicantIncome', 'LoanAmount', 
                 'Loan_Amount_Term', 'Credit_History']])))

# now want to scale all features to same normalisation. Use MinMaxScaler from sklearn.
min_max = MinMaxScaler() # python is so smart.
# scale training and test feature values.
X_train_minmax=min_max.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_minmax=min_max.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
# now apply same 5NN clustering on scaled data and check accuracy.
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_minmax,Y_train.values.ravel())
print "Accuracy_score of 5NN with normalisation = ", float(accuracy_score(Y_test,knn.predict(X_test_minmax)))
""" 
wanted to output the change I made but subtracting two accuracy scores doesn't work at
all for some reason, as in is a factor of two +/- a bit wrong. Need to look into this more 
it's weird because accuracy_score() does return a float and the error is so huge I cant
see it being a floating point error.

print "Accuracy has changed by ", (accuracy_score(Y_test,knn.predict(X_test_minmax)) - 
accuracy_score(Y_test, knn.predict(X_test[['ApplicantIncome' ,'CoapplicantIncome', 'LoanAmount', 
                 'Loan_Amount_Term', 'Credit_History']])))*100, "% due to normalisation."
""" 

