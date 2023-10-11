

import re
import nltk
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore", category=DeprecationWarning)



train = pd.read_csv(r'data\train_E6oV3lV.csv')
test = pd.read_csv(r'data\test_tweets_anuFYb8.csv')


# Check out first 10 non racist/sexist tweets
train[train['label'] == 0].head(10)

# Check first 10 racist/sexist tweets
train[train['label'] == 1].head(10)

# There are quite a few words and characters that are not really required.
# So we will try to keep only those words which are important and add value.

# Let's check dimensions of the train and test datasets.

train.shape, test.shape
# ((31962, 3), (17197, 2))

# Training set has 31,962 tweets and the test set has 17,197.

# Let's look at the label-distribution in the training set.

train["label"].value_counts()

# label
# 0    29720
# 1     2242

# In the training set, about 7% of tweets are labeled as racist or sexist, and 93% are labeled as non racist/sexist.
# We have an imbalanced classification challenge.

length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()
plt.hist(length_train, bins=20, label="train_tweets")
plt.hist(length_test, bins=20, label="test_tweets")
plt.legend()
plt.show()


# Cleaning raw data enables us to get rid of unwanted words and characters which helps in obtaining better features.
# The objective is to clean out noise (items less relevant to finding the sentiment of tweets such as punctuation, special characters, numbers, and terms
# which don't carry much weightage in context to the text.)

# First, we combine datasets to make it more convenient to preprocess the data. We'll split them back into training and testing later.

combi = pd.concat([train, test])
combi.shape

# (49159, 3)

# Create user-defined function to remove unwanted text patterns from the tweets.

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        return input_txt
    

# We'll follow these steps to clean the raw tweets in our data.

# 1. Remove twitter handles as they are already masked as @user for privacy concerns.
# 2. Remove punctuations, numbers, special characters.
# 3. Remove smaller words as they do not add much value.
# 4. Normalize the text data, i.e. reducing words to their root word.

# 1. removing twitter handles

combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\W]*")
combi.head()

# 2. removing punctuations, numbers, special characters

combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
combi.head(10)

# 3. removing short words

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([W for w in x.split() if len(w)>3]))
combi.head()

# 4. text normalization

tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing

tokenized_tweet.head()


# Normalize the tokenized tweets

from nltk.stem.porter import *
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming

# Now we stitch the tokens back together using nltk's MosesDetokenizer function

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    combi['tidy_tweet'] = tokenized_tweet