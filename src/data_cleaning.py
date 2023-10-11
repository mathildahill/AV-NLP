

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



train = pd.read_csv('data/train_E6oV31V.csv')
test = pd.read_csv('data/test_tweets_anuFYb8.csv')