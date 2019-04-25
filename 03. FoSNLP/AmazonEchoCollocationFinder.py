# INFO:
# In this code we find useful collocation bigrams in our Amazon Echo reviews
# 

# PACKAGES ====================================================================
import pandas as pd
import nltk
# DATA ========================================================================

# NLP for amazon review data
data = pd.read_csv("C:/Users/Roland/Documents/git_repositories/NLP/amazon_alexa.tsv", sep = '\t')

# We're only interested in the reviews
data = data['verified_reviews']

# Bring 1000 reviews together:
reviews = ""

for i in range(1000):
    reviews = (reviews + " " + data[i])

# Remove apostrophes:
# reviews = reviews.replace("'", "")
# Convert to lower case:
reviews = reviews.lower()
# Identifying collocations ====================================================

# Initialize our tokenizer to grab words only
tokenizer = nltk.RegexpTokenizer(r'\w+\'?\w+')
# Use our tokenizer to tokenize our review
reviewTokens = tokenizer.tokenize(reviews)

# reviewTokens = nltk.word_tokenize(reviews)

# Retrieve a list of bigrams
tokenBigrams = list(nltk.bigrams(reviewTokens))
# Convert bigrams to strings
tokenStrings = [(bigram[0] + " " + bigram[1]) for bigram in tokenBigrams]
# Initialize empty unique strings list
uniqueStrings = []
# Iterate over strings and keep unique strings
for string in tokenStrings:
    if string not in uniqueStrings:
        uniqueStrings.append(string)

    
# Match strings in text to get frequency count. 
# Result is a list of tuples containg string and frequency
tokenFrequency = [(string, reviews.count(string)) for string in uniqueStrings]

# Sort by frequency
tokenFrequency.sort(key = lambda x:x[1], reverse = True)

# Print top 10:
tokenFrequency[:10]

# We can get the most frequent collocations
# Now we need to filter this list for useful collocations
# That is, collocations of the form ADJ NOUN or NOUN NOUN:
# Let's step back to our bigrams to pull the POS tags for each string
test = tokenBigrams[0] # to be removed
nltk.pos_tag(test) # to be removed

# Create list of text and pos tags
tokenBigramsPOS = [nltk.pos_tag(bigram) for bigram in tokenBigrams]

tokenBigramsPOS[:5]

test = tokenBigramsPOS[0]
test[1][1]

# Look at tag names to find nouns and adjectives
# nltk.help.upenn_tagset()

myAdjectives = ["JJ", "JJR", "JJS"]
myNouns = ["NN", "NNP", "NNPS", "NNS"]

# Filter for collocations of the form mentioned above
usefulBigrams = [bigram for bigram in tokenBigramsPOS if bigram[0][1] in myAdjectives + myNouns and bigram[1][1] in myNouns]

# Convert to tuples containing the strings and the pos tags
usefulStrings = [(bigram[0][0] + " " + bigram[1][0], bigram[0][1] + " " + bigram[1][1]) for bigram in usefulBigrams]

uniqueUsefulStrings = []
# Iterate over strings and keep unique strings
for entry in usefulStrings:
    if entry not in uniqueUsefulStrings:
        uniqueUsefulStrings.append(entry)

# Get frequency
usefulFrequency = [(entry[0], entry[1], reviews.count(entry[0])) for entry in uniqueUsefulStrings]

# Sort by frequency
usefulFrequency.sort(key = lambda x:x[2], reverse = True)

# Print top 10:
usefulFrequency[:10]

# Brilliant! They look like very relevant text for our Amazon Echo reviews!
