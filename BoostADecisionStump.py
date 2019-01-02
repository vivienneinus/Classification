# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 12:05:59 2018

@author: vwzheng
"""
import pandas as pd
import numpy as np
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt      
import os
os.chdir('D:/Downloads/vivienne/ML/Classification_UW')

#Load dataset
loans = pd.read_csv('lending-club-data.csv')

#Extract the target and the feature columns
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.drop('bad_loans', axis=1)
target = 'safe_loans'
loans = loans[[target] + features]
loans = loans.fillna({'emp_length': 'n/a'}) #loans.iloc[122602]

#Transform categorical data into binary features
#Apply one-hot encoding to loans
categorical_variables = []
for feat_name, feat_type in zip(loans.columns, loans.dtypes):
    if feat_type == object:
        categorical_variables.append(feat_name)

for feature in categorical_variables:
    loans_one_hot_encoded = pd.get_dummies(loans[feature],prefix=feature)
    loans_one_hot_encoded.fillna(0)
    loans = loans.drop(feature, axis=1)
    for col in loans_one_hot_encoded.columns:
        loans[col] = loans_one_hot_encoded[col]    

features = list(loans.drop(target, axis=1).columns)

#Load the JSON files for indices
train_idx = pd.read_json('module-8-assignment-2-train-idx.json')
test_idx = pd.read_json('module-8-assignment-2-test-idx.json')
#Train-test split
train_data = loans.iloc[train_idx.iloc[:,0].values]
test_data = loans.iloc[test_idx.iloc[:,0].values]        

#Weighted decision trees
#Write a function to compute weight of mistakes
def intermediate_node_weighted_mistakes(labels_in_node, data_weights):
    #Sum the weights of all entries with label +1
    total_weight_positive = sum(data_weights[labels_in_node == +1])
    
    #Weight of mistakes for predicting all -1's is equal to the sum above
    weighted_mistakes_all_negative = total_weight_positive
    
    #Sum the weights of all entries with label -1
    total_weight_negative = sum(data_weights[labels_in_node == -1])
    
    #Weight of mistakes for predicting all +1's is equal to the sum above
    weighted_mistakes_all_positive = total_weight_negative
    
    #Return the tuple (weight, class_label) representing the lower of the two
    #weights class_label should be an integer of value +1 or -1.
    #If the two weights are identical, 
    if weighted_mistakes_all_negative >= weighted_mistakes_all_positive:
        return (weighted_mistakes_all_positive, +1)
    else:
        return (weighted_mistakes_all_negative, -1)
    
example_labels = np.array([-1, -1, 1, 1, 1])
example_data_weights = np.array([1., 2., .5, 1., 1.])
if intermediate_node_weighted_mistakes(example_labels, example_data_weights) \
   == (2.5, -1):
    print('Test passed!')
else:
    print('Test failed... try again!')
    
example1_labels = np.array([1, 1, 1, 1, 1])
example1_data_weights = np.array([1., 2., .5, 1., 1.])
print(intermediate_node_weighted_mistakes(example1_labels, 
                                          example1_data_weights))    

#Function to pick best feature to split on
#If the data is identical in each feature, this function should return None
def best_splitting_feature(data, features, target, data_weights):
    #These variables will keep track of the best feature and the 
    #corresponding error
    best_feature = None
    best_error = float('+inf') 
    data['data_weights'] = data_weights
    #Loop through each feature to consider splitting on that feature
    for feature in features:
        #Left split will have all data points where the feature value is 0
        #Right split will have all data points where the feature value is 1
        left_split = data[data[feature] == 0]
        right_split = data[data[feature] == 1]
        #Apply the same filtering to data_weights to create left_data_weights,
        #right_data_weights
        left_data_weights = left_split['data_weights']
        right_data_weights = right_split['data_weights']
                    
        #DIFFERENT HERE
        #Calculate the weight of mistakes for left and right sides
        left_weighted_mistakes, left_class = \
        intermediate_node_weighted_mistakes(left_split[target], 
                                            left_data_weights)
        right_weighted_mistakes, right_class = \
        intermediate_node_weighted_mistakes(right_split[target], 
                                            right_data_weights)
        
        #DIFFERENT HERE
        #Compute weighted error by computing
        #([weight of mistakes(left)] + [weight of mistakes(right)]) / 
        #[total weight of all data points]
        error = (left_weighted_mistakes + right_weighted_mistakes) * 1. / \
                sum(data_weights)
        
        #If this is the best error we have found so far, store the feature 
        #and the error
        if error < best_error:
            best_feature = feature
            best_error = error
    
    #Return the best feature we found
    return best_feature    

example2_data_weights = np.array(len(train_data)* [1.5])
#del train_data['data_weights']
if best_splitting_feature(train_data, features, target, 
                          example2_data_weights) == 'term_ 36 months':
    print('Test passed!')
else:
    print('Test failed... try again!')
    
#Build the tree
#Create a leaf node given a set of target values    
def create_leaf(target_values, data_weights):
    #Create a leaf node
    leaf = {'splitting_feature' : None,
            'is_leaf': True}
    
    #Compute weight of mistakes
    #Store the predicted class (1 or -1) in leaf['prediction']
    weighted_error, best_class = \
    intermediate_node_weighted_mistakes(target_values, data_weights)
    leaf['prediction'] = best_class
    
    return leaf

#Learn a weighted decision tree recursively & implement 3 stopping conditions
def weighted_decision_tree_create(data, features, target, data_weights, 
                                  current_depth = 1, max_depth = 10):
    remaining_features = features[:] # Make a copy of the features.
    target_values = data[target]
    print("----------------------------------------------------------------")
    print("Subtree, depth = %s (%s data points)." % (current_depth, 
                                                     len(target_values)))
    
    #Stopping condition 1. Error is 0.
    if intermediate_node_weighted_mistakes(target_values, data_weights)[0] \
        <= 1e-15:
        print("Stopping condition 1 reached.")                
        return create_leaf(target_values, data_weights)
    
    #Stopping condition 2. No more features.
    if remaining_features == []:
        print("Stopping condition 2 reached.")                
        return create_leaf(target_values, data_weights)    
    
    #Additional stopping condition (limit tree depth)
    if current_depth > max_depth:
        print("Reached maximum depth. Stopping for now.")
        return create_leaf(target_values, data_weights)
    
    #If all the datapoints are the same, splitting_feature will be None. 
    #Create a leaf
    splitting_feature = best_splitting_feature(data, features, target, 
                                               data_weights)
    remaining_features.remove(splitting_feature)
        
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    left_data_weights = data_weights[data[splitting_feature] == 0]
    right_data_weights = data_weights[data[splitting_feature] == 1]
    
    print("Split on feature %s. (%s, %s)" % (splitting_feature, 
                                             len(left_split), 
                                             len(right_split)))
    
    #Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(left_split[target], data_weights)
    if len(right_split) == len(data):
        print("Creating leaf node.")
        return create_leaf(right_split[target], data_weights)
    
    #Repeat (recurse) on left and right subtrees
    left_tree = weighted_decision_tree_create(left_split, remaining_features,
                                              target, left_data_weights,    
                                              current_depth + 1, max_depth)
    right_tree = weighted_decision_tree_create(right_split, 
                                               remaining_features, target, 
                                               right_data_weights, 
                                               current_depth + 1, max_depth)
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}

#Count the nodes in your tree    
def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])    

#Make predictions with a weighted decision tree
#Start at the root and traverse down the decision tree in recursive fashion
def classify(tree, x, annotate = False):   
    # If the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print("At leaf, predicting %s" % tree['prediction'])
        return tree['prediction'] 
    else:
        # Split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print("Split on %s = %s" % (tree['splitting_feature'], 
                                        split_feature_value))
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)    

#Evaluate the tree    
def evaluate_classification_error(tree, data):
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x), axis = 1)
    
    # Once you've made the predictions, calculate the classification error
    return sum(prediction != data[target]) / float(len(data))  

#Train a weighted decision
#Suppose we only care about making good predictions for the first 10 &
#last 10 items in train_data, so assign 1 to the first & last 10 items
#0 to the rest.
# Assign weights
example3_data_weights = np.array([1.] * 10 + [0.]*(len(train_data) - 20) \
                                 + [1.] * 10)
# Train a weighted decision tree model.
small_data_decision_tree_subset_20 = weighted_decision_tree_create(
                                     train_data, features, target,
                                     example3_data_weights, max_depth=2)   
del train_data['data_weights']
subset_20 = train_data.head(10).append(train_data.tail(10))
CE_subset_20 = evaluate_classification_error(
               small_data_decision_tree_subset_20, subset_20)
CE_train = evaluate_classification_error(small_data_decision_tree_subset_20, 
                                         train_data)    
'''
The model small_data_decision_tree_subset_20 performs a lot better on 
subset_20 than on train_data. This means:
-The points with higher weights are the ones that are more important during 
the training process of the weighted decision tree.
-The points with zero weights are basically ignored during training.
''' 
#Train a decision tree with only the 20 data points with non-zero weights 
#from the set of points in subset_20 --> get the same model as
#samll_data_decision_tree_subset_20
example_20_data_weights = np.array([1.] * 10 + [1.] * 10)
example_20_decision_tree = weighted_decision_tree_create(
                           subset_20, features, target,
                           example_20_data_weights, max_depth=2)

#Implement Adaboost (on decision stumps)
from math import log
from math import exp
'''
-Stump weights(w_hat) and data point weights(alpha) are two different 
concepts.
-Stump weights tell you how important each stump is while making predictions 
with the entire boosted ensemble.
-Data point weights tell you how important each data point is while training
a decision stump.
'''
def adaboost_with_tree_stumps(data, features, target, num_tree_stumps):
    # start with unweighted data
    alpha = np.array([1.]*len(data))
    weights = []
    tree_stumps = []
    target_values = data[target]
    
    for t in range(num_tree_stumps):
        print('=====================================================')
        print('Adaboost Iteration %d' % t)
        print('=====================================================')      
        # Learn a weighted decision tree stump. Use max_depth=1
        tree_stump = weighted_decision_tree_create(data, features, target, 
                                                   data_weights=alpha, 
                                                   max_depth=1)
        tree_stumps.append(tree_stump)
        
        # Make predictions
        predictions = data.apply(lambda x: classify(tree_stump, x), axis = 1)
        
        # Produce a Boolean array indicating whether
        # each data point was correctly classified
        is_correct = predictions == target_values
        is_wrong   = predictions != target_values
        
        # Compute weighted error
        weighted_error = sum(is_wrong*alpha)/sum(alpha)
        
        # Compute model coefficient using weighted error
        weight = 1./2 * log((1-weighted_error)/weighted_error)
        weights.append(weight)
        
        # Adjust weights on data point
        adjustment = is_correct.apply(lambda is_correct: exp(-weight) 
                                      if is_correct else exp(weight))
        
        # Scale alpha by multiplying by adjustment
        # Then normalize data points weights 
        alpha = alpha * adjustment
        alpha = alpha/sum(alpha)
    
    return weights, tree_stumps

#test run code
small_stump_weights, small_tree_stumps = adaboost_with_tree_stumps(
                                         train_data, features, target, 
                                         num_tree_stumps=2)    
def print_stump(tree):
    split_name = tree['splitting_feature'] 
    #split_name is something like 'term_ 36 months'
    if split_name is None:
        print("(leaf, label: %s)" % tree['prediction'])
        return None
    split_feature, split_value = split_name.split('_')
    print( '                       root')
    print( '         |---------------|----------------|')
    print( '         |                                |')
    print( '         |                                |')
    print( '         |                                |')
    print( '  [{0} == 0]{1}[{0} == 1]    '.format(split_name, 
          ' '*(27-len(split_name))))
    print( '         |                                |')
    print( '         |                                |')
    print( '         |                                |')
    print( '    (%s)                 (%s)' \
          % (('leaf, label: ' + str(tree['left']['prediction']) 
              if tree['left']['is_leaf'] else 'subtree'),
             ('leaf, label: ' + str(tree['right']['prediction']) 
              if tree['right']['is_leaf'] else 'subtree')))
    
print_stump(small_tree_stumps[0])
print_stump(small_tree_stumps[1])

#Train a boosted ensemble of 10 stumps
stump_weights, tree_stumps = adaboost_with_tree_stumps(train_data, features, 
                                                       target, 
                                                       num_tree_stumps=10)

#Make predictions with adaboost
def predict_adaboost(stump_weights, tree_stumps, data):
    scores = np.array([0.]*len(data))
    for i, tree_stump in enumerate(tree_stumps):
        #Compute the predictions using the t-th decision tree
        predictions = data.apply(lambda x: classify(tree_stump, x), axis = 1)
        #Accumulate predictions on scores array
        #Multiply the stump_weights with the predictions
        #Sum the weighted predictions over each stump in the ensemble.
        scores = scores + stump_weights[i] * predictions  
        
    return scores.apply(lambda score : +1 if score > 0 else -1)

predictions = predict_adaboost(stump_weights, tree_stumps, test_data)
accuracy = sum(test_data[target] == predictions) / float(len(predictions))
#0.6203145196036192

#stump_weights are neither monotonically decreasing or increasing
#[0.15802933659263743,
# 0.09311888971129693,
# 0.07288885525840554,
# 0.06706306914118143,
# 0.06456916961644447,
# 0.05456055779178564,
# 0.04351093673362621,
# 0.02898871150041245,
# 0.02596250969152032]
plt.plot(stump_weights)
plt.show()

#Performance plots
#First train an ensemble with num_tree_stumps = 30
stump_weights_30, tree_stumps_30 = adaboost_with_tree_stumps(
                                   train_data, features, target, 
                                   num_tree_stumps = 30)

#Compute training error at the end of each iteration
error_all = []
for n in range(1, 31):
    predictions = predict_adaboost(stump_weights_30[:n], tree_stumps_30[:n], 
                                   train_data)
    error = sum(train_data[target] != predictions)/float(len(predictions))
    error_all.append(error)
    print("Iteration %s, training error = %s" % (n, error_all[n-1]))

#Visualize training error vs number of iterations    
plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1,31), error_all, '-', linewidth=4.0, label='Training error')
plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.legend(loc='best', prop={'size':15})
plt.rcParams.update({'font.size': 16})

#Evaluation on the test data
error_test = []
for n in range(1, 31):
    predictions = predict_adaboost(stump_weights_30[:n], tree_stumps_30[:n], 
                                   test_data)
    error = sum(test_data[target] != predictions)/float(len(predictions))
    error_test.append(error)
    print("Iteration %s, training error = %s" % (n, error_test[n-1])) 

os.chdir('D:/Downloads/vivienne/ML/Classification_UW/Wk5_Boosting')    
plt.rcParams['figure.figsize'] = 7, 5
plt.plot(range(1,31), error_all, '-', linewidth=4.0, label='Training error')
plt.plot(range(1,31), error_test, '-', linewidth=4.0, label='Test error')
plt.title('Performance of Adaboost ensemble')
plt.xlabel('# of iterations')
plt.ylabel('Classification error')
plt.rcParams.update({'font.size': 16})    
plt.legend(loc='best', prop={'size':15})
plt.tight_layout()
plt.savefig('adaboost.png')
#From this plot (with 30 trees), there is no massive overfitting as the # of 
#iterations increases.
