# Import pandas for data work =================================================
import pandas as pd

# Read the data ===============================================================
data = pd.read_csv("C:/Users/Roland/Documents/git_repositories/NLP/airline_tweets_train.csv")

# Examine the data ============================================================
data.head()
data.info()

# Select the columns we want (the text and the sentiment for sentiment analysis)
data = data[["text","airline_sentiment"]]

# Rename sentiment column for ease of typing
data.columns = ["text", "sentiment"]

# Take another look
data.head()

# What types of sentiment do we have?
data.sentiment.unique()

# Plot the distribtuion of tweet sentiment
data.sentiment.value_counts().plot.bar()
# many more negative reviews (as expected)

# Split data into text and sentiment
X = data["text"]
y = data["sentiment"]

# Grab test tweet
test = X[13]


# Building a predictive model =================================================

# Check for null values
data.isnull().sum()

# One null value
data[data["text"].isnull()]

# Nothing to impute here, we'll remove it
data = data.dropna()

# Import STOP_WORDS ===========================================================
from spacy.lang.en.stop_words import STOP_WORDS
# Build a list of stopwords
stopwords = list(STOP_WORDS)
# Take a look
stopwords[:10]

# Import Punctuation ==========================================================
import string
punctuations = string.punctuation
punctuations[:10]


# Text processing =============================================================

# Import spacy
import spacy

# Initialize a natural language processor
nlp = spacy.load("en_core_web_sm")

# Process our text
doc = nlp(test)

# Take a look
print(doc)

# lemma values for each token
[(token.text, token.lemma_, token.pos_) for token in doc]

# Lemmatize non-pronouns and non-hashtags
[token.lemma_.lower().strip() if token.lemma_ != "-PRON-" and token.lemma_ != "#" else token.lower_ for token in doc]
 
# Remove punctuation and stopwords
[token for token in doc if token.is_punct == False and token.is_stop == False] 


# Create spacy parser =========================================================
from spacy.lang.en import English
parser = English()

def spacy_tokenizer(tweet):
    # parse our tweet - this will give us our tokens and lemmas (but not our POS)
    tokens = parser(tweet)

    # lemmatize tokens
    #tokens = [token.lemma_.lower().strip() if token.lemma_ != "-PRON-" and token.lemma_ != "#" else token.lower_ for token in doc]
    tokens = [token.lemma_.lower().strip() if token.lemma_ != "-PRON-" else token.lower_ for token in tokens]
    # Remove punctuation and stopwords
 
    tokens = [token for token in tokens if token not in stopwords and token not in punctuations] 
    # Return our tokens
    return tokens

doc = spacy_tokenizer(test)

# Transformer =================================================================
from sklearn.base import TransformerMixin

# Custom transformer using spacy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]
    def fit(self, X, y = None, **fit_params):
        return self
    def get_params(self, deep = True):
        return {}

# Define clean_text function    
def clean_text(text):
    return text.strip().lower()    

# Vectorizer ==================================================================
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range = (1,1))

    
# Classifier ==================================================================
from sklearn.svm import SVC

# Create Pipeline =============================================================
from sklearn.pipeline import Pipeline

steps = [("cleaner", predictors()),
         ("vectorizer", vectorizer),
         ("SVM", SVC())]

pipe = Pipeline(steps)

# Train Test Split ============================================================
from sklearn.model_selection import train_test_split

X = data["text"]
y = data["sentiment"]

X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size = 0.3, random_state = 27)

# Grid Search CV ==============================================================
from sklearn.model_selection import GridSearchCV
import numpy as np

parameters = {'SVM__C':np.arange(100, 1000, 100),
              'SVM__kernel':["linear", "rbf"],
              'SVM__gamma':["scale"]}

cv = GridSearchCV(pipe, parameters, cv = 3)

# Fit our data ================================================================
cv.fit(X_train, y_train)

# Parameters ==================================================================
print(cv.best_params_)

# Accuracy ====================================================================

# Training Accuracy
print("Accuracy: {}".format(cv.score(X_train, y_train)))

# Testing Accuracy
print("Accuracy: {}".format(cv.score(X_test, y_test)))

# Predictions =================================================================
predictions = cv.predict(X_test)

np.unique(predictions, return_counts = True)

# Confusion matrix ============================================================
from sklearn.metrics import confusion_matrix
# Build the confusion matrix
confusion_matrix(y_test, predictions)

# Let's write our own reviews and see how the model does ======================

my_positive_tweet = ["I had a great flight with @VirginAmerica!"]
my_negative_tweet = ["I had an awful flight with @VirginAmerica!"]

# Predict
print("Positive Tweet Predicted: {} \nNegative Tweet Predicted: {}".format(cv.predict(my_positive_tweet), cv.predict(my_negative_tweet)))
# It gets the good flight review but it labels the bad flight review as neutral.

# Testing a MultinomialNB approach ============================================
from sklearn.naive_bayes import MultinomialNB

# Create a new pipeline with MNB
stepsNB = [("cleaner", predictors()),
         ("vectorizer", vectorizer),
         ("MNB", MultinomialNB())]

pipeNB = Pipeline(stepsNB)

parametersNB = {"MNB__alpha":np.arange(1,10,1)}


cvNB = GridSearchCV(pipeNB, parametersNB, cv = 5)

cvNB.fit(X_train, y_train)

cvNB.best_params_
print("Best score for SVM: {}".format(cvNB.best_score_))
print("Best score for NB: {}".format(cv.best_score_))


# Using a TFIDF Vector ========================================================
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizertfidf = TfidfVectorizer(tokenizer = spacy_tokenizer, ngram_range = (1,1))

# New steps
stepstfidf = [("cleaner", predictors()),
         ("vectorizer", vectorizertfidf),
         ("SVM", SVC())]
# New pipeline
pipetfidf = Pipeline(stepstfidf)

# New parameters
parameterstfidf = {'SVM__C':np.arange(100, 1000, 100),
                   'SVM__kernel':["linear", "rbf"],
                   'SVM__gamma':["scale"]}
# Grid Search
cvtfidf = GridSearchCV(pipetfidf, parameterstfidf, cv = 5)

cvtfidf.fit(X_train, y_train)

print("TFIDF Training Score: {}".format(cvtfidf.best_score_))
print("TFIDF Parameters: {}".format(cvtfidf.best_params_))
print("TFIDF Test Score: {}".format(cvtfidf.score(X_test, y_test)))



# Playground ==================================================================
    
# Custom models class
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import TransformerMixin

# Custom transformer using spacy
class models(TransformerMixin):
    
    def __init__(self, models, params):
        # Once initialized, calling my_models.svm should give my SVM classifier
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.fits = {}
    
    def fit(self, X, y, cv = 3):
        # Iterate through each key (each model)
        for key in self.keys:
            # Get model and parameters
            model = self.models[key]
            params = self.params[key]
            # Initialize grid search instance for current model and its parameters
            gridcv = GridSearchCV(model, params, cv = 3)
            # Fit data to model
            gridcv.fit(X,y)
            # Build dictionary
            self.fits[key] = gridcv
          
    def predict(self, X):
        """
        Returns predictions for each model provided in the model dictionary.
        """
        # Initialize predictions dictionary
        predictions = {}
        # Iterate through each key (each model)
        for key in self.keys:
            # Get fitted model
            model = self.fits[key]
            # Add response predictions to predictions dictionary                      
            predictions[key] = model.predict(X)
        # Create dataframe of predictions
        output = pd.DataFrame(predictions)
        # Return output dataframe
        return output
    
    def score(self, X, y):
        """
        Returns accuracy scores for each model provided in the model dictionary.
        """
        # Initialize scores dictionary    
        scores = {}    
        # Iterate through each key (each model)
        for key in self.keys:
            # Get fitted model
            model = self.fits[key]
            # Get accuracy for current model
            scores[key] = model.score(X,y)
        # Create dataframe of accuracy scores                   
        output = pd.DataFrame(scores, index = [0])
        # Return dataframe of accuracy scores
        return output

# Models
models1 = {"SVM":SVC(),
           "LOGREG":LogisticRegression(),
           "NaiveBayes":MultinomialNB()}

params1 = {"SVM":{"C":np.arange(1,10,1),
                  "kernel":["linear", "rbf"]},
           "LOGREG":{"penalty":["l1","l2"]},
           "NaiveBayes":{"alpha":np.arange(1,10,1)}}


# New steps
steps = [("cleaner", predictors()),
         ("vectorizer", vectorizertfidf),
         ("classifiers", models(models1, params1))]
# New pipeline
pipeline = Pipeline(steps)

# Fit pipeline to data
pipeline.fit(X_train, y_train)

# Check scores
for model in models1.keys():
    print(model + ": " + str(pipeline["classifiers"].fits[model].best_score_))

# Get test scores for each model
scores = pipeline.score(X_test, y_test)




