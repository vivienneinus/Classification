# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:06:10 2018

@author: vwzheng
"""

import pandas as pd
import numpy as np
import os
os.chdir('D:/Downloads/vivienne/ML/Classification_UW')

#Load dataset
loans = pd.read_csv('lending-club-data.csv')

#Explore some features
loans.columns

#Modify the target column
#reassign the labels to have +1 for a safe loan, and -1 for a risky/bad loan
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', axis = 1)

#Select features
target = 'safe_loans'
features = ['grade',                #grade of the loan (categorical)
            'sub_grade_num',   #sub-grade of the loan as a number from 0 to 1
            'short_emp',            #one year or less of employment
            'emp_length_num',       #number of years of employment
            'home_ownership',   #home_ownership status: own, mortgage or rent
            'dti',                  #debt to income ratio
            'purpose',              #the purpose of the loan
            'payment_inc_ratio',    #ratio of the monthly payment to income
            'delinq_2yrs',          #number of delinquincies
            'delinq_2yrs_zero',    #no delinquincies in last 2 years
            'inq_last_6mths',  #number of creditor inquiries in last 6 months
            'last_delinq_none',     #has borrower had a delinquincy
            'last_major_derog_none',#has borrower had 90 day or worse rating
            'open_acc',             #number of open credit accounts
            'pub_rec',              #number of derogatory public records
            'pub_rec_zero',         #no derogatory public records
            'revol_util',           #percent of available credit being used
            'total_rec_late_fee',   #total late fees received to day
            'int_rate',             #interest rate of the loan
            'total_rec_int',        #interest received to date
            'annual_inc',           #annual income of borrower
            'funded_amnt',          #amount committed to the loan
            'funded_amnt_inv',   #amount committed by investors for the loan
            'installment',          #monthly payment owed by the borrower
           ]

#Skip observations with missing values
loans = loans[[target] + features].dropna()

#Apply one-hot encoding to loans
categorical_variables = []
for feat_name, feat_type in zip(loans.columns, loans.dtypes):
    if feat_type == object:
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    loans_one_hot_encoded = pd.get_dummies(loans[feature],prefix=feature)
    loans = pd.concat([loans, loans_one_hot_encoded],axis=1)
    loans = loans.drop(feature, axis=1)

#Import indices of train valid
train_idx = pd.read_json('module-8-assignment-1-train-idx.json')
valid_idx = pd.read_json('module-8-assignment-1-validation-idx.json')
#Split data into training and validation
train_data = loans.iloc[train_idx.iloc[:,0].values]
valid_data = loans.iloc[valid_idx.iloc[:,0].values]

#Gradient boosted tree classifier
import sklearn
from sklearn.ensemble import GradientBoostingClassifier
train_target = train_data[target].as_matrix()
train_features = train_data.drop(target, axis=1).as_matrix()
model_5 = GradientBoostingClassifier(max_depth=6, n_estimators=5).fit(
                                     train_features, train_target)

#Make predictions
valid_safe_loans = valid_data[valid_data[target] == 1]
valid_risky_loans = valid_data[valid_data[target] == -1]

sample_valid_data_risky = valid_risky_loans[0:2]
sample_valid_data_safe = valid_safe_loans[0:2]

sample_valid_data = sample_valid_data_safe.append(sample_valid_data_risky)
sample_valid_data

#Prediction Classes
sample_predictions = model_5.predict(sample_valid_data.drop(target, axis=1
                                                            ).as_matrix())
#prediction accuracy
sample_accuracy = sum(sample_predictions == sample_valid_data[target]) / \
                  len(sample_predictions)
                  
#Prediction Probabilities
sample_predProbas = model_5.predict_proba(sample_valid_data.drop(
                                          target, axis=1).as_matrix())[:,1]
#return the probabilities of being a safe loan
idx_min = np.argmin(sample_predProbas) + 1
#return the loan in sample that is least likely to be a safe loan
#all the predictions with probability >= 0.5, the model predicts: label +1

#Evaluate the model on the validation data
#class predictions
valid_predictions = model_5.predict(valid_data.drop(target, axis=1
                                                    ).as_matrix())
#calculate prediction accuracy
valid_accuracy = sum(valid_predictions == valid_data[target]) / \
                 len(valid_predictions) #.6612

#Calculate the number of false positives
valid_fp = sum((valid_predictions == 1)&(valid_data[target] == -1)) #1654
#Calculate the number of false negatives        
valid_fn = sum((valid_predictions == -1)&(valid_data[target] == 1)) #1491

#Comparison with decision trees
#the prediction accuracy of the decision trees was around 0.6361

'''
As we explored in the decision tree assignment, we calculated the cost of 
the mistakes made by the model. We again consider the same costs as follows:

False negatives: Assume a cost of $10,000 per false negative.
False positives: Assume a cost of $20,000 per false positive.

Assume that the number of false positives and false negatives for 
the learned decision tree was:
    
False negatives: 1936
False positives: 1503
'''
cost_dt = 10000 * 1936  + 20000 * 1503 #49,420,000
cost_gb = 10000 * valid_fn + 20000 * valid_fp #47,990,000

#Most positive & negative loans
#probability predictions for all the loans in validation
valid_predProbas = model_5.predict_proba(valid_data.drop(
                                         target, axis=1).as_matrix())[:,1]
#add probability predictions as a column called predictions into validation
valid_data['predictions'] = valid_predProbas
#Sort the data (in descreasing order) by the probability predictions
valid_data = valid_data.sort_values(by = 'predictions', ascending = False)
#For each row, the probabilities should be a number in the range [0, 1]
#Find the top 5 loans with the highest probability of being a safe loan
print(valid_data.head(5))
#What grades are the top 5 loans?
print(valid_data.head(5)[['grade_A','grade_B', 'grade_C', 'grade_D', 
                          'grade_E', 'grade_F', 'grade_G']])
#find the 5 loans with the lowest probability of being a safe loan
print(valid_data.tail(5)) #last is the least
#valid_data.sort_values(by='predictions', ascending=True).head(5)
print(valid_data.tail(5)[['grade_A','grade_B', 'grade_C', 'grade_D', 
                          'grade_E', 'grade_F', 'grade_G']])

valid_target = valid_data[target].as_matrix()
valid_features = valid_data.drop([target, 'predictions'], axis=1).as_matrix()   

#Effects of adding more trees
model_10 = GradientBoostingClassifier(max_depth=6, n_estimators=10).fit(
                                      train_features, train_target)
accuray_10 = sum(model_10.predict(valid_features) == valid_target) / \
             len(valid_target) #0.66619991383024557
model_50 = GradientBoostingClassifier(max_depth=6, n_estimators=50).fit(
                                      train_features, train_target)
accuray_50 = sum(model_50.predict(valid_features) == valid_target) / \
             len(valid_target) #0.68364928909952605
model_100 = GradientBoostingClassifier(max_depth=6, n_estimators=100).fit(
                                       train_features, train_target)
accuray_100 = sum(model_100.predict(valid_features) == valid_target) / \
              len(valid_target) #0.68968117190866007
model_200 = GradientBoostingClassifier(max_depth=6, n_estimators=200).fit(
                                       train_features, train_target)
accuray_200 = sum(model_200.predict(valid_features) == valid_target) / \
              len(valid_target) #0.68957345971563977
model_500 = GradientBoostingClassifier(max_depth=6, n_estimators=500).fit(
                                       train_features, train_target)    
accuray_500 = sum(model_500.predict(valid_features) == valid_target) / \
              len(valid_target) #0.68634209392503231

#simpler coding style              
train_errors = [] #[0.33450656922539568, 0.32832692979392242, 
#0.28367231790214675, 0.25379510465085042, 0.21497084822268198, 
#0.13458179961847438]
valid_errors = []
#[0.33864713485566567, 0.33380008616975443, 0.31635071090047395, 
#0.31031882809133993, 0.31042654028436023, 0.31365790607496769]
x = [5, 10, 50, 100, 200, 500]
for i in x:
    model = GradientBoostingClassifier(max_depth=6, n_estimators=i).fit(
                                       train_features, train_target)
    accuracy = model.score(valid_features, valid_target)
    classification_error = 1 - accuracy
    valid_errors.append(classification_error)
    train_errors.append(1 - model.score(train_features, train_target))              

#model_100 has the best accuracy on the validation_data?
#it is not always true that the model with the most trees will perform best
#on test data?
           
#Plot the training and validation error vs. number of trees  
#classification error = 1 - accuracy 
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt              

def make_figure(dim, title, xlabel, ylabel, legend):
    plt.rcParams['figure.figsize'] = dim
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(loc=legend, prop={'size':15})
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()              

#Make plot    
plt.plot(x, train_errors, linewidth=4.0, label='Training error')
plt.plot(x, valid_errors, linewidth=4.0, label='Validation error')

make_figure(dim=(10,5), title='Error vs number of trees',
            xlabel='Number of trees',
            ylabel='Classification error',
            legend='best')

os.chdir('D:/Downloads/vivienne/ML/Classification_UW/Wk5_Boosting')
plt.savefig('TrainErrVSValidErr.png')    

#the training error reduces as the number of trees increases
#it is not always true that the validation error will reduce as
#the number of trees increases
              