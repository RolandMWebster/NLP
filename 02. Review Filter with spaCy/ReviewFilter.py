# INFO ========================================================================
# 
# OUTLINE: 
# This code outlines the build of a review search function that aims to
# return the n most informative reviews with respect to a given search
# query.
#    
# NEXT STEPS:
# 1. Consider removing the pulling of proper nouns, this is perhaps not as useful
#    as I'd thought they might be.
# 2. Find a way to pull anonyms for a more representative filtering of reviews.
#    I.e. searching for "good" will also match with reviews that contain "bad".
# 3. Consider looking for objects in the search query that are next to our 
#    adjectives. The idea being that if "speaker" is right next to "good", we 
#    are more than likely asking whether the speaker is good.
#
# CONS: 
# 1. Slow to cycle through all the reviews
# 2. A simple regex model would probably perform just as well. However the
#    framework allows us to increase the complexity to where we may arrive at
#    a more useful search filter.

# PACKAGE IMPORTS =============================================================
import pandas as pd
import spacy
from spacy.lang.en import English
from spacy.matcher import Matcher

# DATA ========================================================================

# NLP for amazon review data
data = pd.read_csv("C:/Users/Roland/Documents/git_repositories/NLP/amazon_alexa.tsv", sep = '\t')

# Let's look at the columns in the data
data.columns

# We're only interested in the reviews and the ratings
data = data[['verified_reviews','rating']]

data.head(n = 5)
# HYPOTHETICAL: Let's say we are interested in buying the Amazon Echo but we've
# heard some concerns from our friends that the sound quality isn't very good.
# Let's see if we can filter reviews based on a question to give us a list of 
# reviews that might help us.
    
# WORKFLOW:
# 1. Pull important words from our search string
# 2. Match the important words in our reviews
# 3. Pull reviews ordered my match count.

# Search string:
searchString = "Is the sound quality of Amazon Echo good?"

# We want to pull important words from our search query. Usually people are asking
# about features or qualities of a product when they search. More formally, we're
# looking for descriptive words, or adjectives. Proper nouns are also useful as it
# narrows down what we're asking the question about:
nlp = spacy.load("en_core_web_sm")

doc = nlp(searchString)

# Now we'll create a list of patterns that we can use to count matches in each of our reviews.
# We'll start by just using our words as patterns.
# We'll use LEMMA to get the base representation of our adjectives, but the names
# of objects or products are frequently recycled other words, such as Echo, we don't
# want to return "echoed" or "echoing" if someone searches about the Amazon Echo.
# We'll use lower for PROPN to take care of capitalization:
# adjPatterns = [[{'LEMMA' : token.text}] for token in doc if token.pos_ == "ADJ"]

# propnPatterns = [[{'LOWER' : token.text}] for token in doc if token.pos_ == "PROPN"]

# adjPatterns = [[{'LEMMA' : token.text},{'POS':'NOUN', 'OP' : '?'}] for token in doc if token.pos_ == "ADJ"]

# queryPattern1 = [{'POS' : 'ADJ'}]
# queryPattern2 = [{'POS' : 'ADJ'}, {'POS':'NOUN', 'OP' : '?'}]
# queryPattern3 = [{'POS':'NOUN', 'OP' : '?'}, {'POS' : 'ADJ'}]

queryPattern = [{'POS':'NOUN', 'OP' : '?'}, {'POS' : 'ADJ'}, {'POS':'NOUN', 'OP' : '?'}]

# queryPatterns = [queryPattern1, queryPattern2, queryPattern3]

matcher = Matcher(nlp.vocab)

# for pattern in queryPatterns: matcher.add("pattern", None, pattern)

matcher.add("QUERY_PATTERN", None, queryPattern)

matches = matcher(doc)  

# iterate over our match object to create a span object
for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)

# Now we can tokenize these words and create patterns as combinations of these words:
keyWords = []

for match_id, start, end in matches:
    matched_span = doc[start:end]
    keyWords = keyWords + [str(matched_span)]

# We receive nested lists. We want to create bigrams of all the words found (regardless
# of where/how they were found). So first we need to split our current bigrams into
# single words:
    
# Split bigrams
keyWords = [keyword.split(" ") for keyword in keyWords]

# Join to 1 list of matched words
individualKeyWords = []
# Merge:
for i in keyWords:
    individualKeyWords = individualKeyWords + i


# Now we have our matched words. We can create bigrams as all possible combinations
# of these words:
def pairWords(words):
    """ Iterates through lists of keywords to create pairs of those keywords in a
        new list object. """
    # Intialize bigrams list:
    bigrams = []     
    # Iterate through all words. Importantly we grab our first word for the bigram here
    for firstword in words:
        # Construct a list of remaining words:
        remainingwords = [word for word in words if word != firstword]
        # Iterate through remaining words to construct bigrams
        for secondword in remainingwords:
            bigrams.append([firstword, secondword])
# Loop throguh each pattern, if there is a match then return the search structure
    return(bigrams)


pairWords(individualKeyWords)
# We want to rewrite how we pull words from our query
# Let's look for adjectives and nouns that surround the adjectives

print(adjPatterns, propnPatterns)

# Now we want to create combinations of pairs of adjectives and nouns, this will
# give us patterns such as "echo sound" or "sound good" to hunt for also:
def pairPatterns(patterns1, patterns2):
    """ Iterates through lists of keywords to create pairs of those keywords in a
        new list object. """
    # Iterate through our adjectives
    for keyword1 in patterns1:
        # Iterate through our proper nouns:
        for keyword2 in patterns2:
            # Return our pairs of keywords:
            yield ([keyword1[0], keyword2[0]])

# Now let's use our function to get all our pairs of keywords: 
adjpropnPatterns = [i for i in pairPatterns(adjPatterns, propnPatterns)]
propnadjPatterns = [i for i in pairPatterns(propnPatterns, adjPatterns)]
adjadjPatterns = [i for i in pairPatterns(adjPatterns, adjPatterns)]
propnpropnPatterns = [i for i in pairPatterns(propnPatterns, propnPatterns)]

# Now we can bring our patterns together
allPatterns = adjPatterns + propnPatterns + adjpropnPatterns + propnadjPatterns + adjadjPatterns + propnpropnPatterns

# Let#s take a look:
# allPatterns

# Now we can initialize our matcher and add our patterns to it:
matcher = Matcher(nlp.vocab)
# Add our patterns:
for pattern in allPatterns: matcher.add("pattern", None, pattern)


# Now we need to loop through our patterns and loop through our reviews and tally
# the number of matches to each review. We'll cherry pick a test review for now:
review = data["verified_reviews"][3]

# Count number of matches per review:

testreviews = data["verified_reviews"][:100]


def getReviewMatches(data, matcher):
    for review in data:
        doc = nlp(review)
        matches = matcher(doc)
    
        yield([review, len(matches)])
        
# Get our number of matches        
results = [i for i in getReviewMatches(testreviews, matcher)]
    
# Take the top 10 reviews and show them to the searcher:
results.sort(key = lambda x: x[1], reverse = True) 

results = results[:10]

for result in results: print(result[0], "\n")


# Immediately I can think of a couple of things going wrong here.
# 1. Our individual has searched for "good" and so our filter has returned
# only the positive sound reviews and isn't even looking for the bad reviews.
# Really, someone that searches "is the sound good?" and someone else who asks
# "is the sound bad?" are askign the same, more generalised question: "what 
# quality is the sound?". It's not a fair representation if we simply show
# all the good reviews.

# Before we hack it apart let's put this all into a function and play around
# with reading some more questions into the filter:

def reviewSearch(query, reviews, n=10):
    """ Queries a review data set using a user provided text query and returns the 10
        most relevent reviews. """
    # Initialize our natural language processor
    nlp = spacy.load("en_core_web_sm")
    # Tokenize our query
    doc = nlp(query)
    # Pull the adjectives and proper nouns
    adjPatterns = [[{'LEMMA' : token.text}] for token in doc if token.pos_ == "ADJ"]
    propnPatterns = [[{'LOWER' : token.text}] for token in doc if token.pos_ == "PROPN"]
    # Print our patterns
    print(adjPatterns, propnPatterns)
    # Create paired patterns
    adjpropnPatterns = [i for i in pairPatterns(adjPatterns, propnPatterns)]
    propnadjPatterns = [i for i in pairPatterns(propnPatterns, adjPatterns)]
    adjadjPatterns = [i for i in pairPatterns(adjPatterns, adjPatterns)]
    propnpropnPatterns = [i for i in pairPatterns(propnPatterns, propnPatterns)]
    # Bring all our patterns together
    allPatterns = adjPatterns + propnPatterns + adjpropnPatterns + propnadjPatterns + adjadjPatterns + propnpropnPatterns
    # Initialize our matcher
    matcher = Matcher(nlp.vocab)
    # Add our patterns
    for pattern in allPatterns: matcher.add("pattern", None, pattern)
    # Get number of matches using getReviewMatches function
    results = [i for i in getReviewMatches(reviews, matcher)]
    # Sort results for most relevent
    results.sort(key = lambda x: x[1], reverse = True) 
    # Take top n results
    results = results[:n]
    # Return results
    #return [result[0] for result in results]
    return results    

# Now we can use our function:
results = reviewSearch("Does it ship internationally?", testreviews, n = 10)
for result in results: print(result, "\n")