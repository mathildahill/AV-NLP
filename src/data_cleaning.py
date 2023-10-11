

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


train[train['label'] == 0].head(10)