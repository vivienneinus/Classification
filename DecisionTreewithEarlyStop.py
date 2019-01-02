# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 02:36:28 2018

@author: vwzheng
"""

import pandas as pd
import numpy as np
import os
os.chdir('D:/Downloads/vivienne/ML/Classification_UW')

#Load the Lending Club dataset
data = pd.read_csv('lending-club-data.csv')
loans = data
#reassign the labels to have +1 for a safe loan, and -1 for a risky/bad loan
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', axis = 1)

#consider these four features
features = ['grade',            #grade of the loan
            'term',             #the term of the loan
            'home_ownership',   #home_ownership status: own, mortgage or rent
            'emp_length',       #number of years of employment
           ]
target = 'safe_loans'

loans = loans[features + [target]]
loans.info()
loans.describe(include='all')
loans.describe(include='object')
loans.grade.unique()
loans.term.unique()
loans.home_ownership.unique()
loans.emp_length.unique()
loans = loans.fillna({'emp_length': 'n/a'}) #loans.iloc[122602]

#Apply one-hot encoding to loans
categorical_variables = []
for feat_name, feat_type in zip(loans.columns, loans.dtypes):
    if feat_type == object:
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    loans_one_hot_encoded = pd.get_dummies(loans[feature],prefix=feature)
    loans = loans.drop(feature, axis=1)
    for col in loans_one_hot_encoded.columns:
        loans[col] = loans_one_hot_encoded[col]

#Load the JSON files for indices
train_idx = pd.read_json('module-6-assignment-train-idx.json')
valid_idx = pd.read_json('module-6-assignment-validation-idx.json')
#Split data into training and validation
train_data = loans.iloc[train_idx.iloc[:,0].values]
valid_data = loans.iloc[valid_idx.iloc[:,0].values]
print(len(train_data))
print(len(valid_data))
print(len(loans.dtypes))

#Early stopping methods for decision trees
#Early stopping condition 1: Maximum depth
#Early stopping condition 2: Minimum node size
def reached_minimum_node_size(data, min_node_size):
    #Return True if the number of data points is less than or equal to 
    #the minimum node size.
    if len(data) <= min_node_size:
        return True
    

#Early stopping condition 3: Minimum gain in error reduction
def error_reduction(error_before_split, error_after_split):
    #Return the error before the split minus the error after the split.
    return (error_before_split - error_after_split)

#Calculate the number of misclassified examples when predicting majority class   
def intermediate_node_num_mistakes(labels_in_node):
    #Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0    
    #Count the number of 1's (safe loans)
    loans_safe = sum(labels_in_node == 1)   
    #Count the number of -1's (risky loans)
    loans_risky = sum(labels_in_node == -1)                
    #Return the number of mistakes that the majority classifier makes.
    return min(loans_safe, loans_risky)   

#Find the best feature to split on given the data and a list of features to 
#consider
def best_splitting_feature(data, features, target):

    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    #Since error is always <= 1, 
    #Intialize it with something larger than 1.

    #Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    #Loop through each feature to consider splitting on that feature
    for feature in features:
        
        #left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        #right split will have all data points where the feature value is 1
        right_split = data[data[feature] == 1]
            
        #Calculate the number of misclassified examples in the left split.
        left_mistakes = intermediate_node_num_mistakes(left_split[target])            

        #Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split[target])
            
        #Compute the classification error of this split.
        #Error = (# of left mistakes+# of right mistakes)/(# of data points)
        error = (left_mistakes + right_mistakes)/num_data_points

        #If this is the best error we have found so far, 
        #store the feature as best_feature and the error as best_error
        if error < best_error:
            best_feature = feature
            best_error = error
    
    return best_feature # Return the best feature we found    
        
#Create a leaf node given a set of target values
def create_leaf(target_values):    
    #Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf': True}
   
    #Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])    

    #For the leaf node, set the prediction to be the majority class.
    #Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1         

    # Return the leaf node
    return leaf    

#Incorporating new early stopping conditions in binary decision tree 
#implementation
def decision_tree_create(data, features, target, current_depth = 0, 
                         max_depth = 10, min_node_size=1, 
                         min_error_reduction=0.0):
    
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print("----------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, 
                                                     len(target_values)))
    
    
    #Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print("Stopping condition 1 reached. All data points have the same \
              target value.")                
        return create_leaf(target_values)
    
    #Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print("Stopping condition 2 reached. No remaining features.")                
        return create_leaf(target_values)    
    
    #Early stopping condition 1: Reached max depth limit.
    if current_depth >= max_depth:
        print("Early stopping condition 1 reached. Reached maximum depth.")
        return create_leaf(target_values)
    
    #Early stopping condition 2: Reached the minimum node size.
    #If the number of data points is less than or equal to the minimum size, 
    #return a leaf.
    if len(data) <= min_node_size:
        print("Early stopping condition 2 reached. Reached minimum node \
              size.")
        return create_leaf(target_values)
    
    #Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)
    
    #Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    #Early stopping condition 3: Minimum error reduction
    #Calculate the error before splitting (number of misclassified examples 
    #divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) \
                         / float(len(data))
    
    #Calculate the error after splitting (number of misclassified examples 
    #in both groups divided by the total number of examples)
    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))
    
    #If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, 
    #return a leaf.
    if  error_reduction(error_before_split, error_after_split) <= \
        min_error_reduction:
        print("Early stopping condition 3 reached. Minimum error reduction.")
        return create_leaf(target_values)
    
    
    remaining_features.remove(splitting_feature)
    print("Split on feature %s. (%s, %s)" % (splitting_feature, 
                                             len(left_split), 
                                             len(right_split)))
    
    
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target,
                                     current_depth + 1, max_depth, 
                                     min_node_size, min_error_reduction)        
    
    ## YOUR CODE HERE
    right_tree = decision_tree_create(right_split, remaining_features, 
                                      target, current_depth + 1, max_depth, 
                                      min_node_size, min_error_reduction)
    
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}
       
def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])   

#features = train_data.drop(target, axis=1).columns #return pandas index
features = list(train_data.drop(target, axis=1).columns) #return list    

small_decision_tree = decision_tree_create(train_data, features, 
                                           'safe_loans', max_depth = 2, 
                                           min_node_size = 10, 
                                           min_error_reduction=0.0)
if count_nodes(small_decision_tree) == 7:
    print('Test passed!')
else:
    print('Test failed... try again!')
    print('Number of nodes found                :', 
          count_nodes(small_decision_tree))
    print('Number of nodes that should be there : 5')
    
#Build a tree
my_decision_tree_new = decision_tree_create(train_data, features, 
                                            'safe_loans', max_depth = 6, 
                                            min_node_size = 100, 
                                            min_error_reduction=0.0) 

#Train a tree model ignoring early stopping conditions 2 and 3
my_decision_tree_old = decision_tree_create(train_data, features, 
                                            'safe_loans', max_depth = 6,  
                                            min_node_size = 0, 
                                            min_error_reduction=-1)



#Making predictions
#Classify a new point x using a given tree
def classify(tree, x, annotate = False):
       #if the node is a leaf node.
    if tree['is_leaf']:
        if annotate:
             print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction']
    else:
        #split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate:
             print("Split on %s = %s" % (tree['splitting_feature'], 
                                         split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)

#consider the first example of the validation set
print(valid_data.iloc[0])
print('Predicted class: %s ' % classify(my_decision_tree_new, 
                                        valid_data.iloc[0]))
#add some annotations to prediction to see what the prediction path was that 
#lead to this predicted class
classify(my_decision_tree_new, valid_data.iloc[0], annotate = True)  
#the prediction path for the decision tree old
classify(my_decision_tree_old, valid_data.iloc[0], annotate = True)          

#Evaluate the model
def evaluate_classification_error(tree, data):
    #Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x), axis = 1)
    #calculate the classification error and return it
    return sum(prediction.values != data[target])*1./len(data)
#Evaluate the classification error of my_decision_tree_new on validation_set
evaluate_classification_error(my_decision_tree_new, valid_data)
#Evaluate the classification error of my_decision_tree_old on validation_set
evaluate_classification_error(my_decision_tree_old, valid_data)    

#Explore the effect of max_depth
#model_1: max_depth = 2 (too small)
model_1 = decision_tree_create(train_data, features, 'safe_loans', 
                               max_depth = 2, min_node_size = 0,
                               min_error_reduction=-1)
#model_2: max_depth = 6 (just right)
model_2 = decision_tree_create(train_data, features, 'safe_loans', 
                               max_depth = 6, min_node_size = 0, 
                               min_error_reduction=-1)
#model_3: max_depth = 14 (may be too large)
model_3 = decision_tree_create(train_data, features, 'safe_loans', 
                               max_depth = 14, min_node_size = 0, 
                               min_error_reduction=-1)

#Evaluate the models on the train and validation data
print("Training data, classification error (model 1):", \
      evaluate_classification_error(model_1, train_data))
print("Training data, classification error (model 2):", \
      evaluate_classification_error(model_2, train_data))
print("Training data, classification error (model 3):", \
      evaluate_classification_error(model_3, train_data))
print("Training data, classification error (model 1):", \
      evaluate_classification_error(model_1, valid_data))
print("Training data, classification error (model 2):", \
      evaluate_classification_error(model_2, valid_data))
print("Training data, classification error (model 3):", \
      evaluate_classification_error(model_3, valid_data))

#Measure the complexity of the tree
#complexity(T) = number of leaves in the tree T
#count the number of leaves in a tree
def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

print("Number of nodes (model 1):", count_leaves(model_1))
print("Number of nodes (model 2):", count_leaves(model_2))
print("Number of nodes (model 3):", count_leaves(model_3))

#Explore the effect of min_error
#model_4: min_error_reduction = -1 (ignoring this early stopping condition)
model_4 = decision_tree_create(train_data, features, 'safe_loans', 
                               max_depth = 6, min_node_size = 0,
                               min_error_reduction=-1)
#model_5: min_error_reduction = 0 (just right)
model_5 = decision_tree_create(train_data, features, 'safe_loans', 
                               max_depth = 6, min_node_size = 0, 
                               min_error_reduction=0)
#model_6: min_error_reduction = 5 (too positive)
model_6 = decision_tree_create(train_data, features, 'safe_loans', 
                               max_depth = 6, min_node_size = 0, 
                               min_error_reduction=5)
print("Validation data, classification error (model 4):", \
      evaluate_classification_error(model_4, valid_data))
print("Validation data, classification error (model 5):", \
      evaluate_classification_error(model_5, valid_data))
print("Validation data, classification error (model 6):", \
      evaluate_classification_error(model_6, valid_data))

print("Number of nodes (model 4):", count_leaves(model_4))
print("Number of nodes (model 5):", count_leaves(model_5))
print("Number of nodes (model 6):", count_leaves(model_6))

#Explore the effect of min_node_size
#model_7: min_node_size = 0 (too small)
model_7 = decision_tree_create(train_data, features, 'safe_loans', 
                               max_depth = 6, min_node_size = 0,
                               min_error_reduction=-1)
#model_8: min_node_size = 2000 (just right)
model_8 = decision_tree_create(train_data, features, 'safe_loans', 
                               max_depth = 6, min_node_size = 2000,
                               min_error_reduction=-1)
#model_9: min_node_size = 50000 (too large)
model_9 = decision_tree_create(train_data, features, 'safe_loans', 
                               max_depth = 6, min_node_size = 50000,
                               min_error_reduction=-1)

print("Validation data, classification error (model 7):", \
      evaluate_classification_error(model_7, valid_data))
print("Validation data, classification error (model 8):", \
      evaluate_classification_error(model_8, valid_data))
print("Validation data, classification error (model 9):", \
      evaluate_classification_error(model_9, valid_data))

print("Number of nodes (model 7):", count_leaves(model_7))
print("Number of nodes (model 8):", count_leaves(model_8))
print("Number of nodes (model 9):", count_leaves(model_9))