# Import pandas for data work
import pandas as pd

# Read the data
data = pd.read_csv("C:/Users/Roland/Documents/git_repositories/NLP/airline_tweets_train.csv")

# Examine the data
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
from sklearn.feature_extraction.text import CountVectorizers

vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range = (1,1))

    
# Classifier ==================================================================
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

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
from sklearn.grid_search import GridSearchCV
import numpy as np

parameters = {'SVM__C':np.arange(100, 1000, 100),
              'SVM__kernel':["linear", "rbf"]}

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


# Let's write our own reviews and see how the model does ======================

my_positive_tweet = ["I had a great flight with @VirginAmerica!"]
my_negative_tweet = ["I had an awful flight with @VirginAmerica!"]

# Predict
print("Positive Tweet Predicted: {} \nNegative Tweet Predicted: {}".format(cv.predict(my_positive_tweet), cv.predict(my_negative_tweet)))


print(cv.predict(my_negative_tweet))
# It gets the good flight review but it labels the bad flight review as neutral.
# Maybe the word "awful" isn't used very often over here in the states! Let's swap
# out our words.
my_positive_tweet = ["I had a delightful flight with @VirginAmerica!"]
my_negative_tweet = ["I had a really terrible flight with @VirginAmerica! I'll never fly with them again!"]

print("Positive Tweet Predicted: {} \nNegative Tweet Predicted: {}".format(cv.predict(my_positive_tweet), cv.predict(my_negative_tweet)))

# Interesting!
