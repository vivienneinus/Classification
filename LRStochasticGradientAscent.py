# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:20:51 2018

@author: vwzheng
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('D:/Downloads/vivienne/ML/Classification_UW')

#Load and process review dataset
products = pd.read_csv('amazon_baby_subset.csv')

#Apply text cleaning on the review data
import json
with open('important_words.json') as important_words_file:    
    important_words = json.load(important_words_file)

#Remove punctuation
products = products.fillna({'review': ''}) #fill in N/A's in the review col
def remove_punctuation(text):
    import string
    tr = str.maketrans("", "", string.punctuation)
    return text.translate(tr)
products['review_clean'] = products['review'].apply(remove_punctuation)
    
#Compute word counts for important_words
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : 
                                                    s.split().count(word))
        
#Count reviews containing "perfect" from important_words    
products['contains_perfect'] = products['perfect'].apply(lambda x: 1 if x >=1
                                                         else 0)  
count_perfect = sum(products['contains_perfect'])       

#Split data into training and validation sets
with open('module-10-assignment-train-idx.json') as train_idx_file:    
    train_idx = json.load(train_idx_file) 
with open('module-10-assignment-validation-idx.json') as valid_idx_file:    
    valid_idx = json.load(valid_idx_file)     
train_data = products.iloc[train_idx]    
valid_data = products.iloc[valid_idx] 

#Convert data frame to multi-dimensional array
def get_numpy_data(dataframe, features, label):
    dataframe['constant'] = 1
    features = ['constant'] + features
    features_frame = dataframe[features]
    feature_matrix = features_frame.as_matrix()
    label_sarray = dataframe[label]
    label_array = label_sarray.as_matrix()
    return (feature_matrix, label_array)

feature_matrix_train, sentiment_train = get_numpy_data(train_data, 
                                                       important_words, 
                                                       'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(valid_data,
                                                       important_words, 
                                                       'sentiment')

#use stochastic gradient ascent to train the classifier using logistic 
#regression. changing the solver to stochastic gradient ascent does not 
#affect the number of features: 194 including constant

#Build on logistic regression
def predict_probability(feature_matrix, coefficients):
    #Take dot product of feature_matrix and coefficients  
    score = np.dot(feature_matrix, coefficients)
    #Compute P(y_i = +1 | x_i, w) using the link function
    predictions = 1./(1+np.exp(-score))
    #return predictions
    return predictions

#Derivative of log likelihood with respect to a single coefficient
def feature_derivative(errors, feature):     
    #Compute the dot product of errors and feature
    derivative = np.dot(errors, feature)
    #Return the derivative
    return derivative    

#To verify the correctness of the gradient computation, provide a function 
#for computing average log likelihood
def compute_avg_log_likelihood(feature_matrix, sentiment, coefficients):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    logexp = np.log(1. + np.exp(-scores))
    #Simple check to prevent overflow
    mask = np.isinf(logexp)
    logexp[mask] = -scores[mask]   
    
    lp = np.sum((indicator-1)*scores - logexp)/len(feature_matrix)   
    
    return lp
#ll_A(w) = 1/N * ll(w)
    
#Modify the derivative for stochastic gradient ascent
j = 1                        # Feature number
i = 10                       # Data point number
coefficients = np.zeros(194) # A point w at which are computing gradient

predictions = predict_probability(feature_matrix_train[i:i+1,:], 
                                  coefficients)
indicator = (sentiment_train[i:i+1]==+1)

errors = indicator - predictions
gradient_single_data_point =feature_derivative(errors, 
                                               feature_matrix_train[i:i+1,j])
print("Gradient single data point: %s" % gradient_single_data_point)
print("           --> Should print 0.0")    
#the code block computes a scalar

#Modify the derivative for using a batch of data points
j = 1                        # Feature number
i = 10                       # Data point start
B = 10                       # Mini-batch size
coefficients = np.zeros(194) # A point w at which are computing the gradient.

predictions = predict_probability(feature_matrix_train[i:i+B,:], 
                                  coefficients)
indicator = (sentiment_train[i:i+B]==+1)

errors = indicator - predictions
gradient_mini_batch = feature_derivative(errors, 
                                         feature_matrix_train[i:i+B,j])
print("Gradient mini-batch data points: %s" % gradient_mini_batch)
print("                --> Should print 1.0")
#the code block computes a scalar
#The value of batch size is the length of data (len(train_data)) 47780,
#the batch gradient ascent acts as the same as full gradient

#Average the gradient across a batch
#a common practice to normalize the gradient update rule by the batch size B

#Implement stochastic gradient ascent
#The function should return the final set of coefficients, along with the 
#list of log likelihood values over time
'''
* Create an empty list called log_likelihood_all
* Initialize coefficients to initial_coefficients
* Set random seed = 1
* Shuffle the data before starting the loop below
* Set i = 0, the index of current batch

* Run the following steps max_iter times, performing linear scans over the 
  data:
  * Predict P(y_i = +1|x_i,w) using your predict_probability() function
    Make sure to slice the i-th row of feature_matrix with [i:i+batch_size,:]
  * Compute indicator value for (y_i = +1)
    Make sure to slice the i-th entry with [i:i+batch_size]
  * Compute the errors as (indicator - predictions)
  * For each coefficients[j]:
    - Compute the derivative for coefficients[j] and save it to derivative.
      Make sure to slice the i-th row of feature_matrix with 
      [i:i+batch_size,j]
    - Compute the product of the step size, the derivative, and 
      (1./batch_size).
    - Increment coefficients[j] by the product just computed.
  * Compute the average log likelihood over the current batch.
    Add this value to the list log_likelihood_all.
  * Increment i by batch_size, indicating the progress made so far on the 
    data.
  * Check whether we made a complete pass over data by checking
    whether (i+batch_size) exceeds the data size. If so, shuffle the data. 
    If not, do nothing.

* Return the final set of coefficients, along with the list 
  log_likelihood_all.
'''
def logistic_regression_SG(feature_matrix, sentiment, initial_coefficients, 
                           step_size, batch_size, max_iter):
    log_likelihood_all = []

    #make sure it's a numpy array
    coefficients = np.array(initial_coefficients)
    #set seed=1 to produce consistent results
    np.random.seed(seed=1)
    #Shuffle the data before starting
    permutation = np.random.permutation(len(feature_matrix))
    feature_matrix = feature_matrix[permutation,:]
    sentiment = sentiment[permutation]

    i = 0 # index of current batch
    # Do a linear scan over data
    for itr in range(max_iter):
        #Predict P(y_i = +1|x_i,w) using your predict_probability function
        #Make sure to slice the i-th row of feature_matrix 
        #with [i:i+batch_size,:]
        predictions = predict_probability(feature_matrix[i:i+batch_size,:], 
                                          coefficients)

        #Compute indicator value for (y_i = +1)
        #Make sure to slice the i-th entry with [i:i+batch_size]
        indicator = (sentiment[i:i+batch_size]==+1)

        #Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in range(len(coefficients)): # loop over each coefficient
            #Recall feature_matrix[:,j] is the feature column associated
            #with coefficients[j]
            #Compute the derivative for coefficients[j] and save it to 
            #derivative.
            #Make sure to slice the i-th row of feature_matrix 
            #with [i:i+batch_size,j]
            derivative = feature_derivative(errors, 
                                            feature_matrix[i:i+batch_size,j])
            #Compute the product of the step size, the derivative, and
            #the **normalization constant** (1./batch_size)
            coefficients[j] += (1./batch_size) * step_size* derivative

        #Check whether log likelihood is increasing
        #Print the log likelihood over the *current batch*
        lp = compute_avg_log_likelihood(feature_matrix[i:i+batch_size,:], 
                                        sentiment[i:i+batch_size],
                                        coefficients)
        log_likelihood_all.append(lp)
        if itr <= 15 or (itr <= 1000 and itr % 100 == 0) or \
           (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0 or \
           itr == max_iter-1:
            data_size = len(feature_matrix)
            print('Iteration %*d: Average log likelihood (of data points \
                                                          [%0*d:%0*d]) = \
                   %.8f' % \
                (int(np.ceil(np.log10(max_iter))), itr, \
                 int(np.ceil(np.log10(data_size))), i, \
                 int(np.ceil(np.log10(data_size))), i+batch_size, lp))  

        #if we made a complete pass over data, shuffle and restart
        i += batch_size
        if i+batch_size > len(feature_matrix):
            permutation = np.random.permutation(len(feature_matrix))
            feature_matrix = feature_matrix[permutation,:]
            sentiment = sentiment[permutation]
            i = 0                

    #return the list of log likelihoods for plotting purposes.
    return coefficients, log_likelihood_all

#Checkpoint
sample_feature_matrix = np.array([[1.,2.,-1.], [1.,0.,1.]])
sample_sentiment = np.array([+1, -1])

coefficients_sample, log_likelihood_sample = logistic_regression_SG(
                                                      sample_feature_matrix, 
                                                      sample_sentiment,
                                                      np.zeros(3),
                                                      step_size=1., 
                                                      batch_size=2, 
                                                      max_iter=2)    
print('--------------------------------------------------------------------')
print('Coefficients learned                 :', coefficients_sample)
print('Average log likelihood per-iteration :', log_likelihood_sample)
if np.allclose(coefficients_sample, np.array([-0.09755757,  0.68242552, 
                                              -0.7799831]), atol=1e-3)\
   and np.allclose(log_likelihood_sample, np.array([-0.33774513108142956, 
                                                    -0.2345530939410341])):
    #pass if elements match within 1e-3
    print('----------------------------------------------------------------')
    print('Test passed!')
else:
    print('----------------------------------------------------------------')
    print('Test failed')

#Compare convergence behavior of stochastic gradient ascent
'''
When the value of batch size above is len(train_data)=47780, the stochastic 
gradient ascent function logistic_regression_SG acts as a standard gradient 
ascent algorithm
'''

#Run stochastic gradient ascent: batch_size = 1, Small Caveat
coefficientsS, log_likelihoodS = logistic_regression_SG(
                                 feature_matrix_train, sentiment_train,
                                 np.zeros(194), step_size=5e-1, 
                                 batch_size=1, max_iter=10)  
#as each iteration passes, the average log likelihood fluctuates in the 
#batch change    
plt.plot(log_likelihoodS)
plt.show()

#Run batch gradient ascent: batch_size = len(train_data),
coefficientsBGA, log_likelihoodBGA = logistic_regression_SG(
                                     feature_matrix_train, sentiment_train,
                                     np.zeros(194), step_size=5e-1, 
                                     batch_size = len(train_data), 
                                     max_iter=200)
#as each iteration passes, the average log likelihood increases in the 
#batch change  
plt.plot(log_likelihoodBGA)
plt.show()

#Make "passes" over the dataset
'''
#_passes = #_data points touched so far/size_dataset
'''
#Suppose that we run stochastic gradient ascent with a batch size of 100. 
#1000 gradient updates are performed at the end of two passes over a 
#dataset consisting of 50000 data points b/c #_passes is number to complete 
#the whole dataset, for each batch size we update 1 gradient, so 2*50000/100

#Log likelihood plots for stochastic gradient ascent
num_passes = 10
batch_size = 100
num_iterations = num_passes * int(len(train_data)/batch_size)
#sga with 10 passes
coefficientssga, log_likelihoodsga = logistic_regression_SG(
                                     feature_matrix_train, sentiment_train,
                                     np.zeros(194), step_size=1e-1, 
                                     batch_size=100, max_iter=num_iterations)

#Generate a plot of average log likelihood as a function of number of passes
def make_plot(log_likelihood_all, len_data, batch_size, smoothing_window=1, 
              label=''):
    plt.rcParams.update({'figure.figsize': (9,5)})
    log_likelihood_all_ma = np.convolve(np.array(log_likelihood_all), \
                                        np.ones((smoothing_window,))/\
                                        smoothing_window, mode='valid')

    plt.plot(np.array(range(smoothing_window-1, len(log_likelihood_all)))*\
             float(batch_size)/len_data, log_likelihood_all_ma, 
             linewidth=4.0, label=label)
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
    plt.xlabel('# of passes over data')
    plt.ylabel('Average log likelihood per data point')
    plt.legend(loc='lower right', prop={'size':14})
    
make_plot(log_likelihoodsga, len_data=len(train_data), batch_size=100,
          label='stochastic gradient, step_size=1e-1') 
plt.title('Avg log likelihood vs Number of passes')   

os.chdir('D:/Downloads/vivienne/ML/Classification_UW/Wk7_Scaling&Stochastic\
GradientAscent')
make_plot(log_likelihoodsga, len_data=len(train_data), batch_size=100,
          smoothing_window=30, label='stochastic gradient, step_size=1e-1')
plt.title('Avg log likelihood vs Number of passes') 
plt.savefig('Stochastic Gradient Ascent.png')

#Stochastic gradient ascent vs batch gradient ascent
num_passes = 200
batch_size = 100
num_iterations = num_passes * int(len(train_data)/batch_size)
#SGA with 200 passes
coefficientsSGA, log_likelihoodSGA = logistic_regression_SG(
                                     feature_matrix_train, sentiment_train,
                                     np.zeros(194), step_size=1e-1, 
                                     batch_size=100, max_iter=num_iterations)
make_plot(log_likelihoodSGA, len_data=len(train_data), batch_size=100,
          smoothing_window=30, label='stochastic, step_size=1e-1')
make_plot(log_likelihoodBGA, len_data=len(train_data), 
          batch_size=len(train_data),
          smoothing_window=1, label='batch, step_size=5e-1')
plt.title('Stochastic Gradient Ascent vs Batch Gradient Ascent') 
plt.savefig('SGAvsBGA.png')

#Explore the effects of step sizes on stochastic gradient ascent

num_passes = 10
batch_size = 100
num_iterations = num_passes * int(len(train_data)/batch_size)
#Plot the log likelihood as a function of passes for each step size
coeffsga = []
log_likesga = []
step_sizes = np.logspace(-4, 2, num=7)
for step_size in step_sizes:
    coeff, log_like = logistic_regression_SG(feature_matrix_train, 
                                                   sentiment_train,
                                                   np.zeros(194),
                                                   step_size=step_size, 
                                                   batch_size=batch_size, 
                                                   max_iter=num_iterations)
    coeffsga.append(coeff)
    log_likesga.append(log_like)
    make_plot(log_like, len_data=len(train_data), batch_size=100,
              smoothing_window=30, label='step_size=%.1e'%step_size)
    
plt.title('Stochastic Gradient Ascent vs Step Size') 
plt.savefig('SGAstepsizes.png')    
#100 is the worst step size b/c it results in the lowest log likelihood
#leave the worst step size (1e2=step_sizes[7]) out to find out the best one
for i in range(6):
    make_plot(log_likesga[i], len_data=len(train_data), batch_size=100,
              smoothing_window=30, label='step_size=%.1e'%step_sizes[i])
#1 is the best step size b/c it results in the highest log likelihood    