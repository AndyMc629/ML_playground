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
import numpy as np
#==============================================================================
# KNN modules.
#==============================================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

#==============================================================================
# Logistic Regression modules.
#==============================================================================
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale

#==============================================================================
# Support Vector Machine modules.
#==============================================================================
from sklearn import svm

#==============================================================================
# Read in data.
#==============================================================================

# personal note: I hate it when a tutorial gives you data pre-packaged like this,
# it's sooo unatural .... >.<
X_train = pd.read_csv("../data/loan_prediction/X_train.csv")
Y_train = pd.read_csv("../data/loan_prediction/Y_train.csv")

X_test = pd.read_csv("../data/loan_prediction/X_test.csv")
Y_test = pd.read_csv("../data/loan_prediction/Y_test.csv")

# let's have a look.
plt.figure(1)
X_train[X_train.dtypes[(X_train.dtypes == "float64")|(X_train.dtypes == "int64")]
                        .index.values].hist(figsize=[11, 11])
plt.show()
plt.close()                        
        
#==============================================================================
# Knn regression.                
#==============================================================================

# Currently the numerical data is all over the place in terms of normalisation.
# Let's run a kNN clustering algorithm to see how this impacts the prediction.

# Note: need to ravel() the y_values to stop a data conversion warning (expects a 1d array).
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                 'Loan_Amount_Term', 'Credit_History']], Y_train.values.ravel())

# check performance of model on test set, remember should only ever do this at end
# in a normal analysis!
print "Accuracy_score of 5NN without normalisation = ", \
float(accuracy_score(Y_test, knn.predict(X_test[['ApplicantIncome' ,
                                                 'CoapplicantIncome', 'LoanAmount', 
                                                 'Loan_Amount_Term', 'Credit_History']])))

# now want to scale all features to same normalisation. Use MinMaxScaler from sklearn.
min_max = MinMaxScaler() # python is so smart.

# scale training and test feature values.
X_train_minmax=min_max.fit_transform(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_minmax=min_max.fit_transform(X_test[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
# now apply same 5NN clustering on scaled data and check accuracy.
knn=KNeighborsClassifier(n_neighbors=5) # did I need to re-declare this? 
knn.fit(X_train_minmax,Y_train.values.ravel())
print "Accuracy_score of 5NN with normalisation = ", \
float(accuracy_score(Y_test,knn.predict(X_test_minmax))), '-'*60


""" 
wanted to output the change I made but subtracting two accuracy scores doesn't work at
all for some reason, as in is a factor of two +/- a bit wrong. Need to look into this more 
it's weird because accuracy_score() does return a float and the error is so huge I cant
see it being a floating point error.

print "Accuracy has changed by ", (accuracy_score(Y_test,knn.predict(X_test_minmax)) - 
accuracy_score(Y_test, knn.predict(X_test[['ApplicantIncome' ,'CoapplicantIncome', 
                                           'LoanAmount', 'Loan_Amount_Term', 
                                           'Credit_History']])))*100, "% due to normalisation."
""" 


#==============================================================================
# Logistic Regression
#==============================================================================

# apply logistic reg to unscaled data
logisticReg = LogisticRegression(penalty='l2', C=0.01)

logisticReg.fit(X_train[['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                 'Loan_Amount_Term', 'Credit_History']], Y_train.values.ravel())

print "Accuracy_score of logistic regression without normalisation = ", \
float(accuracy_score(Y_test,logisticReg.predict(X_test[['ApplicantIncome', 
                                                        'CoapplicantIncome', 'LoanAmount', 
                                                        'Loan_Amount_Term', 'Credit_History']])))

# apply logistic reg to scaled data for comparison.
logisticReg = LogisticRegression(penalty='l2', C=0.01)
logisticReg.fit(X_train_minmax, Y_train.values.ravel())
print "Accuracy_score of logistic regression with normalisation = ", \
float(accuracy_score(Y_test,logisticReg.predict(X_test_minmax)))

# As you can see, this normalisation is not useful, we need the data to be standardized
# i.e transformed to be normally distributed around mu=0 with sigma=1.

# Standardizing the train and test data
X_train_scale=scale(X_train[['ApplicantIncome', 'CoapplicantIncome',
                'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
X_test_scale=scale(X_test[['ApplicantIncome', 'CoapplicantIncome',
               'LoanAmount', 'Loan_Amount_Term', 'Credit_History']])
# Fitting logistic regression on our standardized data set
logisticReg=LogisticRegression(penalty='l2',C=0.01)
logisticReg.fit(X_train_scale,Y_train.values.ravel())
# print accuracy
print "Accuracy_score of logistic regression with standardized data = ", \
accuracy_score(Y_test,logisticReg.predict(X_test_scale)), '-'*60

# try a parameter scan for fun.
regularisationParameters = np.linspace(1e-5, 1e5+1e-5, 11)
accuracyscores = []

for Ctest in regularisationParameters:          
    logisticReg=LogisticRegression(penalty='l2',C=Ctest)
    logisticReg.fit(X_train_scale,Y_train.values.ravel())
    accuracyscores.append(accuracy_score(Y_test,logisticReg.predict(X_test_scale)))
# ok so regularisation parameter size really doesn't affect anything in this case 
# i.e all the accuracy scores are exactly the same. Interesting or obvious?
# the smallest value of the reg paramenter C does give a far smaller accuracy
# for the case where we use l1 regularization. 
# see http://cs.nyu.edu/~rostami/presentations/L1_vs_L2.pdf for some more info.
  

"""
Note : "Choosing between scaling and standardizing is a confusing choice, 
you have to dive deeper in your data and learner that you are going to use to 
reach the decision. For starters, you can try both the methods and check cross 
validation score for making a choice." 
"""

#==============================================================================
# Support Vector Machine
#==============================================================================
svm = svm.SVC(C= 1.0, # let's look at all the param's I should be thinking about.
cache_size= 10, 
class_weight= None,
coef0= 0.0,
decision_function_shape= None,
degree= 3,
gamma= 100,
kernel= 'rbf',
max_iter= -1, 
probability= True,
random_state= None,
shrinking= True,
tol= 0.001,
verbose=1)

svm.fit(X_train_scale, Y_train.values.ravel())
print "Accuracy_score of SVM with standardized data = ", \
accuracy_score(Y_test,svm.predict(X_test_scale)), '-'*60