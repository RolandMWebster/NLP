# Building classification model:

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
test = X[1]

# Import spacy for nlp ========================================================
import spacy

# Load our model
nlp = spacy.load("en_core_web_sm")

# Tokenize our test tweet
doc = nlp(test)

# Generate a dataframe view of our tokens and some of their properties
tokenized_text = pd.DataFrame()

for i, token in enumerate(doc):
    tokenized_text.loc[i, 'text'] = token.text
    tokenized_text.loc[i, 'lemma'] = token.lemma_,
    tokenized_text.loc[i, 'pos'] = token.pos_
    tokenized_text.loc[i, 'tag'] = token.tag_
    tokenized_text.loc[i, 'dep'] = token.dep_
    tokenized_text.loc[i, 'shape'] = token.shape_
    tokenized_text.loc[i, 'is_alpha'] = token.is_alpha
    tokenized_text.loc[i, 'is_stop'] = token.is_stop
    tokenized_text.loc[i, 'is_punctuation'] = token.is_punct

tokenized_text


# Add custom is_hashtag and is_profile token attributes =======================
from spacy.tokens import Doc, Token

def get_is_profile(token):
    """ Returns true if the first character of a word is an "at" symbol @. """
    return True if token.text[0] == "@" else False
    
def get_is_hashtag(word):
    """ Returns true if the first character of a word is a hash symbol #. """
    return True if token.text[0] == "#" else False

# Initialize our new token attributes (leaving it here would give us our attributes
# but would read false for all tokens in our document)
Token.set_extension('is_profile', force = True, getter = get_is_profile)
Token.set_extension('is_hashtag', force = True,  getter = get_is_hashtag)

# Re-Tokenize our data
doc = nlp(test)

# Generate a dataframe view of our tokens with their two new attributes
tokenized_text = pd.DataFrame()

# Print our tokens and our boolean for is profile or is hashtag
print([(token.text, token._.is_profile, token._.is_hashtag) for token in doc])

# Add custom entity "recipients" ==============================================
# This will allow us to neatly get the recipient for each tweet
from spacy.matcher import Matcher

# Define our get recipient function that creates a matcher and finds matches
def get_recipients(doc):
    
    # Initialize matcher
    matcher = Matcher(nlp.vocab)
    # Create pattern to match to
    profile_pattern = [{"TEXT": {"REGEX": "@.*"}}]
    # Add pattern to the matcher
    matcher.add("PROFILES", None, profile_pattern)
    # Get matches in doc
    matches = matcher(doc)
    # Initialize empty recipients list
    recipients = []
    # Loop through matches and add recipients
    for match_id, start, end in matches:
        recipients.append(doc[start:end])
           
    # Return our list of recipients    
    return recipients


# Set our new doc extension (we imported doc earlier when we imported Token)
Doc.set_extension("recipients", getter = get_recipients, force = True)

# Re-Tokenize our tweet
doc = nlp(test)

# Print our recipient entity
print("recipients:", doc._.recipient)





