# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 22:36:44 2018

@author: vwzheng
"""
#??If you have Graphviz, go ahead and re-visualize small_model here 
#to do the traversing for this data point.

import pandas as pd
import numpy as np
import os
os.chdir('D:/Downloads/vivienne/ML/Classification_UW')

#Load the Lending Club dataset
#data = pd.read_csv('lending-club-data.csv')
#loans=data
loans = pd.read_csv('lending-club-data.csv')

#Explore some features
print(loans.columns)

#Explore the target column
# safe_loans =  1 => safe
# safe_loans = -1 => risky
loans['safe_loans'] = loans['bad_loans'].apply(lambda x: +1 if x==0 else -1)
del loans['bad_loans']
#loans = loans.drop('bad_loans', axis=1)

#Explore the distribution of the column safe_loans
percent_safe = sum(loans['safe_loans'] == 1)/float(len(loans['safe_loans']))
percent_risky = sum(loans['safe_loans'] == -1)*1./len(loans['safe_loans'])

#Features for the classification algorithm
features = ['grade',                 #grade of the loan
            'sub_grade',             #sub-grade of the loan
            'short_emp',             #one year or less of employment
            'emp_length_num',        #number of years of employment
            'home_ownership',   #home_ownership status: own, mortgage or rent
            'dti',                   #debt to income ratio
            'purpose',               #the purpose of the loan
            'term',                  #the term of the loan
            'last_delinq_none',      #has borrower had a delinquincy
            'last_major_derog_none', #has borrower had 90 day or worse rating
            'revol_util',            #percent of available credit being used
            'total_rec_late_fee',    #total late fees received to day
           ]

target = 'safe_loans' # prediction target (y) (+1 means safe, -1 is risky)

#Extract the feature columns and target column
loans = loans[features + [target]]

'''
#Sample data to balance classes
safe_loans_raw = loans[loans[target] == +1]
risky_loans_raw = loans[loans[target] == -1]
print("Number of safe loans  : %s" % len(safe_loans_raw))
print("Number of risky loans : %s" % len(risky_loans_raw))

#Since there are fewer risky loans than safe loans, find the ratio of the 
#sizesand use that percentage to undersample the safe loans.
percentage = len(risky_loans_raw)/float(len(safe_loans_raw))

risky_loans = risky_loans_raw
safe_loans = safe_loans_raw.sample(frac=percentage, random_state=1)

# Append the risky_loans with the downsampled version of safe_loans
loans_data = risky_loans.append(safe_loans)
print(sum(loans_data['safe_loans']==1)*1./loans_data.shape[0])
'''

#One-hot encoding
#change loans_data to loans as import indices for train-valid split
categorical_variables = []
for feat_name, feat_type in zip(loans.columns, loans.dtypes):
    if feat_type == object:
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    loans_one_hot_encoded = pd.get_dummies(loans[feature],prefix=feature)
    loans = pd.concat([loans, loans_one_hot_encoded],axis=1)
    loans = loans.drop(feature,axis=1)
    #for col in loans_one_hot_encoded.columns:
        #loans[col] = loans_one_hot_encoded[col]

#Split data into training and validation
train_idx = pd.read_json('module-5-assignment-1-train-idx.json')
valid_idx = pd.read_json('module-5-assignment-1-validation-idx.json')
train_data = loans.iloc[train_idx.iloc[:,0].values]
valid_data = loans.iloc[valid_idx.iloc[:,0].values]
    
#Build a decision tree classifier
from sklearn.tree import DecisionTreeClassifier

train_Y = train_data[target].as_matrix()
train_X = train_data.drop(target, axis=1).as_matrix()

decision_tree_model = DecisionTreeClassifier(max_depth=6)
decision_tree_model = decision_tree_model.fit(train_X, train_Y)

small_model = DecisionTreeClassifier(max_depth=2)
small_model = small_model.fit(train_X, train_Y)

#Visualizing a learned model (Optional)   
from io import StringIO
from IPython.display import Image
out = StringIO()
from sklearn import tree
import graphviz
tree.export_graphviz(small_model, out_file=out,
                     feature_names=train_data.drop(target,axis=1).columns,
                     class_names=['+1','-1'])

import os
import sys
def conda_fix(graph):
    path = os.path.join(sys.base_exec_prefix, "Library", "bin", "graphviz")
    paths = ("dot", "twopi", "neato", "circo", "fdp")
    paths = {p: os.path.join(path, "{}.exe".format(p)) for p in paths}
    graph.set_graphviz_executables(paths)

import pydotplus
graph=pydotplus.graph_from_dot_data(out.getvalue())
conda_fix(graph)
Image(graph.create_png())
graph.write_png('simple_tree.png')

#another way to visualize the tree
from sklearn import tree
import graphviz 
from os import system

dot_data=tree.export_graphviz(small_model, out_file='simple_tree.dot',
                              feature_names=train_data.drop(target,
                                                            axis=1).columns,  
                              class_names=['+1','-1'],  
                              filled=True, rounded=True, 
                              special_characters=True) 
system("dot -Tpng simple_tree.dot -o simple_tree.png")

from IPython.display import Image
Image(filename='simple_tree.png')

#Make predictions
#grab 2 positive examples and 2 negative examples
valid_safe_loans = valid_data[valid_data[target] == 1]
valid_risky_loans = valid_data[valid_data[target] == -1]

sample_valid_data_risky = valid_risky_loans.iloc[0:2]
sample_valid_data_safe = valid_safe_loans.iloc[0:2]

sample_valid_data = sample_valid_data_safe.append(sample_valid_data_risky)
sample_valid_data.info()
sample_valid_data.describe(include='all')

sample_valid_Y = sample_valid_data[target].as_matrix()
sample_valid_X = sample_valid_data.drop(target,axis=1).as_matrix()

#Predict whether or not a loan is safe
predict_valid_Y = decision_tree_model.predict(sample_valid_X) 
#[ 1, -1, -1,  1]
#percentage of the correct predictions on sample_validation_data 
percent_correct_valid = (sum(predict_valid_Y==sample_valid_Y)*1./
                         len(sample_valid_Y)) 

#Explore probability predictions
predict_probability = decision_tree_model.predict_proba(sample_valid_X)[:,1]
#[ 0.65843457,  0.46369354,  0.35249042,  0.79210526]
#verify that for all the predictions with probability >= 0.5, 
#the model predicted the label +1
loan_idxmax = np.argmax(predict_probability)+1

#Tricky predictions
predict_probability_small = small_model.predict_proba(sample_valid_X)[:,1]
#[ 0.58103415,  0.40744661,  0.40744661,  0.76879888]
#the probability preditions are the exact same for the 2nd and 3rd loans
#b/c during tree traversal 2nd and 3rd examples fall into the same leaf node

#Visualize the prediction on a tree
sample_valid_data.iloc[1]

#Evaluate accuracy of the decision tree model
accuracy_train_small = sum(small_model.predict(train_X)==train_Y
                           )*1./len(train_Y)
accuracy_train_dt = sum(decision_tree_model.predict(train_X)==train_Y
                        )*1./len(train_Y)
#small_model performs worse than the decision_tree_model on the train set

valid_Y = valid_data[target].as_matrix()
valid_X = valid_data.drop(target,axis=1).as_matrix()
accuracy_valid_small = sum(small_model.predict(valid_X)==valid_Y
                           )*1./len(valid_Y) 
accuracy_valid_dt = sum(decision_tree_model.predict(valid_X)==valid_Y
                        )*1./len(valid_Y) #.6361

#Evaluate accuracy of a complex decision tree model
big_model = DecisionTreeClassifier(max_depth=10)
big_model = big_model.fit(train_X, train_Y)
accuracy_train_big = sum(big_model.predict(train_X)==train_Y)*1./len(train_Y)
accuracy_valid_big = sum(big_model.predict(valid_X)==valid_Y)*1./len(valid_Y) 
#big_model has even better performance on the training set than 
#decision_tree_model did but worse on the validation set --> overfitting

#Quantify the cost of mistakes
predict_valid = decision_tree_model.predict(valid_X)
false_negatives = sum((predict_valid==-1)&(valid_Y==1)) #1715
false_positives = sum((predict_valid==1)&(valid_Y==-1)) #1661
correct_predictions = sum(predict_valid==valid_Y)
cost = 10000*false_negatives + 20000*false_positives