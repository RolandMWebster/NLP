# Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# NLP for amazon review data
data = pd.read_csv("C:/Users/Roland/Documents/git_repositories/NLP/amazon_alexa.tsv", sep = '\t')

# Let's look at the columns in the data
data.columns

# We're only interested in the reviews and the ratings
data = data[['verified_reviews','rating']]

# Let's modify review to be a binary column, either good review (4,5) or bad review (1,2,3)
# Start by building a dictionary to map our rating values
#rating_mapping = {1:"Bad",
#                  2:"Bad",
#                  3:"Bad",
#                  4:"Good",
#                  5:"Good"}

# Use our dictionary to update our ratings column
#data = data.replace({"rating":rating_mapping})

# let's look at the review distribution
#data['rating'].value_counts()

# Class imbalance (much more positive reviews than negative, as expected.
# Undersample our data (remove good reviews) to get a more even split
#bad_review_count = data['rating'].value_counts().loc["Bad"]

#bad_reviews = data[data["rating"] == "Bad"]
#good_reviews = data[data["rating"] == "Good"].sample(n = bad_review_count)

#data = good_reviews.append(bad_reviews)

#data['rating'].value_counts()

# Split the reviews and the ratings
X = data['verified_reviews']
y = data['rating']

# Carry out a train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 27)




# COUNT vectorization ==================================================================================================
count_vectorizer = CountVectorizer(stop_words = "english")

# Turn our train and test reviews into vectors of counts
count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)

# Initiate our Naive-Bayes classifier
nb_classifier = MultinomialNB()

# Fit our classifier
nb_classifier.fit(count_train, y_train)

# Predict our test data
pred = nb_classifier.predict(count_test)

# Test accuracy
countNaiveBayesAccuracy = metrics.accuracy_score(y_test, pred)
countNaiveBayesCM = metrics.confusion_matrix(y_test, pred)

# Get the class labels
class_labels = nb_classifier.classes_

# Extract the features
feature_names = count_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])

# TF-IDF vectorization =================================================================================================
# This is unlikely to perform as well with our Naive-Bayes as the naive-bayes framework expects integer values.
tfidf_vectorizer = TfidfVectorizer(stop_words = "english")

# Turn our train and test reviews into vectors of counts
tfidf_train = tfidf_vectorizer.fit_transform(X_train.values)
tfidf_test = tfidf_vectorizer.transform(X_test.values)

# Initiate our Naive-Bayes classifier
nb_classifier = MultinomialNB()

# Fit our classifier
nb_classifier.fit(tfidf_train, y_train)

# Predict our test data
pred = nb_classifier.predict(tfidf_test)

# Test accuracy
tfidfNaiveBayesAccuracy = metrics.accuracy_score(y_test, pred)
tfidfNaiveBayesCM = metrics.confusion_matrix(y_test, pred)

# Get the class labels
class_labels = nb_classifier.classes_

# Extract the features
feature_names = tfidf_vectorizer.get_feature_names()

# Zip the feature names together with the coefficient array and sort by weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))

# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])

# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])




# LOGISTIC REGRESSION ==================================================================================================

# Logistic regression on count
lgrCount = LogisticRegression(C = 1.0)
lgrCount.fit(count_train, y_train)
lgrPredictions = lgrCount.predict(count_test)
countLogRegAccuracy = lgrCount.score(count_test, y_test)
countLogRegCM = metrics.confusion_matrix(y_test, lgrPredictions)


# Logistic regression on tf-idf
lgrTFIDF = LogisticRegression()
lgrTFIDF.fit(tfidf_train, y_train)
tfidfPredictions = lgrTFIDF.predict(tfidf_test)
tfidfLogRegAccuracy = lgrTFIDF.score(tfidf_test, y_test)
tfidfLogRegCM = metrics.confusion_matrix(y_test, tfidfPredictions)



# Compare accuracy of our models:
accuracies = {"Count Naive Bayes:":countNaiveBayesAccuracy,
              "TF-IDF Naive Bayes:":tfidfNaiveBayesAccuracy,
              "Count Log Reg:":countLogRegAccuracy,
              "TF-IDF Log Reg:":tfidfLogRegAccuracy}

for model in accuracies:
    print(model + str(accuracies[model]))

# The logistic regression model on count vectors is performing the best so far!

# Let's try some different alpha values =======================================

alphas = np.arange(0,1.1,0.1)

# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha = alpha)
    # Fit to the training data
    nb_classifier.fit(count_train, y_train)
    # Predict the labels: pred
    pred = nb_classifier.predict(count_test)
    # Compute accuracy: score
    score = metrics.accuracy_score(y_test, pred)
    return score
    
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()