# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:11:18 2018

@author: vwzheng
"""

import pandas as pd
import numpy as np
import os
os.chdir('D:/Downloads/vivienne/ML/Classification_UW')

#Load and process review dataset
#Load the dataset into a data frame named products
products = pd.read_csv('amazon_baby_subset.csv')

#Remove punctuation
products = products.fillna({'review': ''}) #fill in N/A's in the review col
def remove_punctuation(text):
    import string
    tr = str.maketrans("", "", string.punctuation)
    return text.translate(tr)
products['review_clean'] = products['review'].apply(remove_punctuation) 

#Compute word counts (only for the important_words)
import json
with open('important_words.json') as important_words_file:    
    important_words = json.load(important_words_file)
for word in important_words:
    products[word] = products['review_clean'].apply(lambda s : 
                                                    s.split().count(word))
#Take "perfect" for example        
products['perfect'][:3] #products['perfect'].head(3)

#Train-Validation split
train_idx = pd.read_json('module-4-assignment-train-idx.json')
valid_idx = pd.read_json('module-4-assignment-validation-idx.json')
train_data = products.iloc[train_idx.iloc[:,0].values]
valid_data = products.iloc[valid_idx.iloc[:,0].values]
       
#Convert data frame to multi-dimensional array
path=('D:/Downloads/vivienne/ML/Classification_UW/Wk2_LogisticRegression&L2')
os.chdir(path)
from LogisticRegressionviaGradientAscent import get_numpy_data                                                
                                                
feature_matrix_train, sentiment_train = get_numpy_data(train_data, 
                                                       important_words, 
                                                       'sentiment')
feature_matrix_valid, sentiment_valid = get_numpy_data(valid_data, 
                                                       important_words, 
                                                       'sentiment') 
        
#Build on logistic regression with no L2 penalty assignment  
from LogisticRegressionviaGradientAscent import predict_probability
#Add L2 penalty   
#Add L2 penalty to the derivative   
def feature_derivative_with_L2(errors, feature, coefficient, l2_penalty, 
                               feature_is_constant): 
    
    # Compute the dot product of errors and feature
    ## YOUR CODE HERE
    derivative = np.dot(errors, feature)

    # add L2 penalty term for any feature that isn't the intercept.
    if not feature_is_constant: 
        ## YOUR CODE HERE
        derivative -= 2*coefficient*l2_penalty
        
    return derivative

#Compute log likelihood to verify the correctness of the gradient descent 
#algorithm
def compute_log_likelihood_with_L2(feature_matrix, sentiment, coefficients, 
                                   l2_penalty):
    indicator = (sentiment==+1)
    scores = np.dot(feature_matrix, coefficients)
    
    lp = np.sum((indicator-1)*scores-np.log(1.+ np.exp(-scores))
                )-l2_penalty*np.sum(coefficients[1:]**2)
    
    return lp  

#Fit a logistic regression model under L2 regularization  
def logistic_regression_with_L2(feature_matrix, sentiment, 
                                initial_coefficients, step_size, l2_penalty,
                                max_iter):
    # make sure it's a numpy array
    coefficients = np.array(initial_coefficients) 
    lplist = []
    for itr in range(max_iter):
        # Predict P(y_i = +1|x_i,w) using your predict_probability() 
        #function
        ## YOUR CODE HERE
        predictions = predict_probability(feature_matrix, coefficients)
        
        # Compute indicator value for (y_i = +1)
        indicator = (sentiment==+1)
        
        # Compute the errors as indicator - predictions
        errors = indicator - predictions
        for j in range(len(coefficients)): # loop over each coefficient
            is_intercept = (j == 0)
            # Recall that feature_matrix[:,j] is the feature column 
            #associated with coefficients[j].
            # Compute the derivative for coefficients[j]. Save it in a 
            #variable called derivative
            ## YOUR CODE HERE
            derivative = feature_derivative_with_L2(errors, 
                                                    feature_matrix[:,j], 
                                                    coefficients[j], 
                                                    l2_penalty, 
                                                    is_intercept)
            
            # add the step size times the derivative to the current 
            #coefficient
            ## YOUR CODE HERE
            coefficients[j] += step_size*derivative
        
        # Checking whether log likelihood is increasing
        if itr <= 15 or (itr <= 100 and itr % 10 == 0) or (itr <= 1000 and 
                                                           itr % 100 == 0) \
        or (itr <= 10000 and itr % 1000 == 0) or itr % 10000 == 0:
            lp = compute_log_likelihood_with_L2(feature_matrix, sentiment, 
                                                coefficients, l2_penalty)
            lplist.append(lp)
            print('iteration %*d: log likelihood of observed labels = %.8f'%\
                  (int(np.ceil(np.log10(max_iter))), itr, lp))
    
    import matplotlib.pyplot as plt
    x= [i for i in range(len(lplist))]
    plt.plot(x,lplist,'ro')
    plt.show()        
        
    return coefficients    

#Explore effects of L2 regularization
#Train 6 models with L2 penalty values 0, 4, 10, 1e2, 1e3, and 1e5
coefficients_0_penalty = logistic_regression_with_L2(feature_matrix_train, 
                                                     sentiment_train, 
                                                     np.zeros(194),
                                                     5e-6, 0, 501)

coefficients_4_penalty = logistic_regression_with_L2(feature_matrix_train, 
                                                     sentiment_train, 
                                                     np.zeros(194),
                                                     5e-6, 4, 501)

coefficients_10_penalty = logistic_regression_with_L2(feature_matrix_train, 
                                                      sentiment_train, 
                                                      np.zeros(194),
                                                      5e-6, 10, 501)

coefficients_1e2_penalty = logistic_regression_with_L2(feature_matrix_train, 
                                                       sentiment_train, 
                                                       np.zeros(194),
                                                       5e-6, 1e2, 501)

coefficients_1e3_penalty = logistic_regression_with_L2(feature_matrix_train, 
                                                       sentiment_train, 
                                                       np.zeros(194),
                                                       5e-6, 1e3, 501)

coefficients_1e5_penalty = logistic_regression_with_L2(feature_matrix_train, 
                                                       sentiment_train, 
                                                       np.zeros(194),
                                                       5e-6, 1e5, 501)

#Compare coefficients
df_coefs = pd.DataFrame({'word':important_words,
                         #exclude intercept
                         '0_penalty':list(coefficients_0_penalty[1:]),
                         '4_penalty':list(coefficients_4_penalty[1:]),
                         '10_penalty':list(coefficients_10_penalty[1:]),
                         '1e2_penalty':list(coefficients_1e2_penalty[1:]),
                         '1e3_penalty':list(coefficients_1e3_penalty[1:]),
                         '1e5_penalty':list(coefficients_1e5_penalty[1:])
                         })
 
#sorted in a descending order of the coefficients trained with L2 penalty 0 
word_coef = df_coefs.reindex(columns=['word', '0_penalty', '4_penalty', 
                                      '10_penalty', '1e2_penalty', 
                                      '1e3_penalty', '1e5_penalty'])  
word_coef = word_coef.sort_values(by = '0_penalty', ascending = False)

#5 most positive words (with largest positive coefficients)
positive_words = word_coef['word'][:5]
#5 most negative words (with largest negative coefficients)   
negative_words = word_coef['word'][-5:]  

#%matplotlib inline #not working for spyder
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 10, 6

#Make a plot of the coefficients for the 10 words over the different values 
#of L2 penalty
def make_coefficient_plot(table, positive_words, negative_words, 
                          l2_penalty_list):
    cmap_positive = plt.get_cmap('Reds')
    cmap_negative = plt.get_cmap('Blues')
    
    xx = l2_penalty_list
    plt.plot(xx, [0.]*len(xx), '--', lw=1, color='k')
    
    table_positive_words = table[table['word'].isin(positive_words)]
    table_negative_words = table[table['word'].isin(negative_words)]
    del table_positive_words['word']
    del table_negative_words['word']
    
    for i in range(len(positive_words)):
        color = cmap_positive(0.8*((i+1)/(len(positive_words)*1.2)+0.15))
        plt.plot(xx, table_positive_words.iloc[i:i+1].as_matrix().flatten(),
                 '-', label=positive_words.iloc[i], linewidth=4.0, 
                 color=color)
        
    for i in range(len(negative_words)):
        color = cmap_negative(0.8*((i+1)/(len(negative_words)*1.2)+0.15))
        plt.plot(xx, table_negative_words.iloc[i:i+1].as_matrix().flatten(),
                 '-', label=negative_words.iloc[i], linewidth=4.0, 
                 color=color)
        
    plt.legend(loc='best', ncol=3, prop={'size':16}, columnspacing=0.5)
    plt.axis([1, 1e5, -1, 2])
    plt.title('Coefficient path')
    plt.xlabel('L2 penalty ($\lambda$)')
    plt.ylabel('Coefficient value')
    plt.xscale('log')
    plt.rcParams.update({'font.size': 18})
    plt.tight_layout()

l2_penalty_list=[0, 4, 10, 1e2, 1e3, 1e5]
make_coefficient_plot(word_coef, positive_words, negative_words, 
                      l2_penalty_list)

#Measure accuracy
def model_accuracy(feature_matrix, sentiment, coefficients):
    scores = np.dot(feature_matrix, coefficients)
    threshold = np.vectorize(lambda x: 1 if x>0 else -1)
    class_predictions = threshold(scores)
    accuracy = sum(class_predictions == sentiment)/len(sentiment)
    return accuracy

accuracy_0_train = model_accuracy(feature_matrix_train, sentiment_train,
                                  coefficients_0_penalty)
accuracy_4_train = model_accuracy(feature_matrix_train, sentiment_train,
                                  coefficients_4_penalty) 
accuracy_10_train = model_accuracy(feature_matrix_train, sentiment_train,
                                   coefficients_10_penalty)
accuracy_1e2_train = model_accuracy(feature_matrix_train, sentiment_train,
                                    coefficients_1e2_penalty) 
accuracy_1e3_train = model_accuracy(feature_matrix_train, sentiment_train,
                                    coefficients_1e3_penalty)
accuracy_1e5_train = model_accuracy(feature_matrix_train, sentiment_train,
                                    coefficients_1e5_penalty)    
accuracy_0_valid = model_accuracy(feature_matrix_valid, sentiment_valid,
                                  coefficients_0_penalty)
accuracy_4_valid = model_accuracy(feature_matrix_valid, sentiment_valid,
                                  coefficients_4_penalty) 
accuracy_10_valid = model_accuracy(feature_matrix_valid, sentiment_valid,
                                   coefficients_10_penalty)
accuracy_1e2_valid = model_accuracy(feature_matrix_valid, sentiment_valid,
                                    coefficients_1e2_penalty) 
accuracy_1e3_valid = model_accuracy(feature_matrix_valid, sentiment_valid,
                                    coefficients_1e3_penalty)
accuracy_1e5_valid = model_accuracy(feature_matrix_valid, sentiment_valid,
                                    coefficients_1e5_penalty) 
accuracy = pd.DataFrame({'train': [accuracy_0_train, 
                                   accuracy_4_train, 
                                   accuracy_10_train, 
                                   accuracy_1e2_train,
                                   accuracy_1e3_train, 
                                   accuracy_1e5_train],
                         'valid': [accuracy_0_valid, 
                                   accuracy_4_valid, 
                                   accuracy_10_valid, 
                                   accuracy_1e2_valid,
                                   accuracy_1e3_valid, 
                                   accuracy_1e5_valid]}, 
                         index=l2_penalty_list)
    
plt.rcParams['figure.figsize'] = 6, 6    
position = np.arange(len(accuracy.index))    
plt.plot(position, accuracy['train'], 'ro', position, accuracy['valid'], 
         'bs')
plt.xticks(position, accuracy.index, rotation=90)
plt.title('Training vs Validation')
plt.xlabel('L2 penalty ($\lambda$)')
plt.ylabel('accuracy')
# legend
plt.legend(('train','valid'))
plt.show()  

#return indices of min value among rows in each col 
penalty_train, penalty_valid = accuracy.idxmax(axis=0) 