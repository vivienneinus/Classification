# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:49:26 2018

@author: vwzheng
"""

import pandas as pd
import numpy as np
import os
os.chdir('D:/Downloads/vivienne/ML/Classification_UW')

#Load Amazon dataset
products = pd.read_csv('amazon_baby.csv')

#Perform text cleaning
'''
def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text
products['review_clean'] = products['review'].apply(remove_punctuations)
'''
products['review_clean'] = products['review'].str.replace('[^\w\s]','') 
#fill in N/A's in the review column
products = products.fillna({'review':''})  

#Extract Sentiments
#ignore all reviews with rating = 3, as they tend to have a neutral sentiment
products = products[products['rating'] != 3]

#Assign reviews with a rating of 4 or higher to be positive reviews, while 
#the ones with rating of 2 or lower are negative
products['sentiment'] = products['rating'].apply(lambda rating: 
                                                 +1 if rating > 3 else -1)

#Split into training and test sets    
train_idx = pd.read_json('module-2-assignment-train-idx.json')
test_idx = pd.read_json('module-2-assignment-test-idx.json')
train_data = products.iloc[train_idx.iloc[:,0].values]
test_data = products.iloc[test_idx.iloc[:,0].values]

#Build the word count vector for each review
#bag-of-word features
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
#Use this token pattern to keep single-letter words
#First, learn vocabulary from the training data and assign columns to words
#Then convert the training data into a sparse matrix
#train_data['review_clean'].isnull().sum() > 0 --> there're nulls
train_data = train_data.fillna({'review_clean':''}) #X
train_matrix = vectorizer.fit_transform(train_data['review_clean']) #h(X)
#Second, convert the test data into a sparse matrix, 
#using the same word-column mapping
test_data = test_data.fillna({'review_clean':''})
test_matrix = vectorizer.transform(test_data['review_clean'])

#Train a sentiment classifier with logistic regression
#use the sparse word count matrix (train_matrix) as features and the column 
#sentiment of train_data as the target
from sklearn.linear_model import LogisticRegression
sentiment_model = LogisticRegression()
sentiment_model.fit(train_matrix, train_data['sentiment'])

#positive weights w_j correspond to weights that cause positive sentiment, 
#while negative weights correspond to negative sentiment
#Calculate the number of positive (>= 0, nonnegative) coeffs 
cntnonneg=np.sum(sentiment_model.coef_>=0)+np.sum(sentiment_model.intercept_
                                                  >0)

#Making predictions with logistic regression
#Take the 11th, 12th, and 13th data points in the test data and save them to 
#sample_test_data
sample_test_data = test_data.iloc[10:13]
sample_test_data.iloc[0]['review']
sample_test_data.iloc[1]['review']

#The sentiment_model should predict +1 if the sentiment is positive
#-1 if the sentiment is negative
sample_test_matrix = vectorizer.transform(sample_test_data['review_clean'])
#calculate the score of each data point with decision_function()
scores = sentiment_model.decision_function(sample_test_matrix) #WTransh(X)
print (scores)

#Prediciting Sentiment
#make class predictions from scores
def predictions(scores):
    """ make class predictions
    """
    preds = []
    for score in scores:
        if score > 0:
            pred = 1
        else:
            pred = -1
        preds.append(pred)
    return preds
sample_test_predictions = predictions(scores)
#Use predict() to check
sentiment_model.predict(sample_test_matrix)

#Probability Predictions
#calculate the probability predictions from scores as the probability 
#that a sentiment is positive
def probability(scores):
    probs = []
    for score in scores:
        probs.append(1/(1+np.exp(-score)))
    return probs
sample_test_probabilities = probability(scores)
sentiment_model.predict_proba(sample_test_matrix)   

#Find the most positive (and negative) review
#find the 20 reviews in the test_data with the highest probability of 
#being classified as a positive review 
test_probs = sentiment_model.predict_proba(test_matrix)
#return indices of sorted values in ascending order
positive20 = np.argsort(-test_probs[:,1])[:20]
#find products based on indices
positive20_test = test_data.iloc[positive20]
#find the 20 reviews in the test_data with the lowest probability of 
#being classified as a positive review
negative20 = np.argsort(test_probs[:,1])[:20]
negative20_test = test_data.iloc[negative20]

#Compute accuracy of the classifier
#Use the sentiment_model to compute class predictions.
predictions_test = sentiment_model.predict(test_matrix)
#Count the number of data points when the predicted class labels match the 
#ground truth labels.
cnt_correct = np.sum(predictions_test == test_data['sentiment'])
#Divide the total number of correct predictions by the total number of data 
#points in the dataset.
accuracy_test_sentiment = cnt_correct/len(predictions_test)

#Learn another classifier with fewer words -- 20
significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 
                     'loves', 'well', 'able', 'car', 'broke', 'less', 'even',
                     'waste', 'disappointed', 'work', 'product', 'money', 
                     'would', 'return']

#redo vectorizer with limit to 20 words
vectorizer_subset = CountVectorizer(vocabulary=significant_words) 
train_matrix_subset=vectorizer_subset.fit_transform(train_data['review_clean'
                                                               ])
test_matrix_subset = vectorizer_subset.transform(test_data['review_clean'])

#Train a logistic regression model on a subset of data
simple_model = LogisticRegression()
simple_model.fit(train_matrix_subset, train_data['sentiment'])
#Sort the data frame by the coefficient value in descending order
#Make sure that the intercept term is excluded from this table
#coefs_simple = sorted(simple_model.coef_, reverse=True)
'''
How many of the 20 coefficients (corresponding to the 20 significant_words) 
are positive for the simple_model?
Are the positive words in the simple_model also positive words in the 
sentiment_model?
'''
simple_coefs = pd.DataFrame({'word':significant_words,
                             'coefficient':simple_model.coef_.flatten()})
#simple_coefs df
simple_coefs.sort_values(['coefficient'], ascending=False)
nonneg_simple = np.sum(simple_model.coef_>=0)
significant_words_pos = simple_coefs[simple_coefs['coefficient']>0]['word']

sentiment_coefs = pd.DataFrame({'word':vectorizer.get_feature_names(),
                                'coefficient':sentiment_model.coef_.flatten()
                                })
#subset of words not in significant_words_pos
#sentiment_coefs[~sentiment_coefs.word.isin(significant_words_pos)]    
significant_words_pos_sentiment = sentiment_coefs[sentiment_coefs.word.isin(
                                                  significant_words_pos)]
    
#Comparing models    
#compute the classification accuracy of sentiment_model on train_data
pred_train_sentiment = sentiment_model.predict(train_matrix)    
CA_sentiment = np.sum(pred_train_sentiment==train_data['sentiment'])/len(
               pred_train_sentiment)  
#compute the classification accuracy of the simple_model on the train_data
pred_train_simple = simple_model.predict(train_matrix_subset)  
CA_simple = np.sum(pred_train_simple==train_data['sentiment'])/len(
            pred_train_simple)

pred_test_simple = simple_model.predict(test_matrix_subset)
accuracy_test_simple = np.sum(pred_test_simple==test_data['sentiment'])/len(
                       predictions_test)

#Baseline: Majority class prediction --> fraction of sentiment>0 in dataset
#use the majority class classifier as the a baseline (or reference) model 
pos_test = len(test_data[test_data['sentiment']>0])
neg_test = len(test_data[test_data['sentiment']<0])#len(test_data)-pos_test
majority_accuracy=pos_test/(len(test_data))