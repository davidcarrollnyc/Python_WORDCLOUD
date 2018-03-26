!curl -L -o mydatafile.csv http://azuremlsamples.azureml.net/templatedata/Text%20-%20Input.csv

import os
import pandas as pd

dirname = os.getcwd()

mydata = pd.read_csv("mydatafile.csv", header=0)
print(mydata.shape)

# use 10000 for testing
mydata = mydata[:10000]
mydata.head()

import re


# %% clean data
def clean_text(mystring):
    mystring = re.sub(r"@\w+", "", mystring)  # remove twitter handle
    mystring = re.sub(r"\d", "", mystring)  # remove numbers
    mystring = re.sub(r"_+", "", mystring)  # remove consecutive underscores
    mystring = mystring.lower()  # tranform to lower case

    return mystring.strip()


mydata["tweet_text_cleaned"] = mydata.tweet_text.apply(clean_text)

from nltk.tokenize import RegexpTokenizer

preprocessed = [" ".join(RegexpTokenizer(r'\w+'). \
                         tokenize(mydata.tweet_text_cleaned[idx])) \
                for idx in mydata.index]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

custom_stop_words = []
my_stop_words = text.ENGLISH_STOP_WORDS.union(custom_stop_words)

vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 1),
                             stop_words=my_stop_words)

tfidf = vectorizer.fit_transform(preprocessed)
print("Created document-term matrix of size {} x {}".format(*tfidf.shape[:2]))

from sklearn import decomposition
import numpy as np

nmf = decomposition.NMF(init='nndsvd', n_components=3, max_iter=200)
W = nmf.fit_transform(tfidf)
H = nmf.components_
print("Generated factor W of size {} and factor H of size {}".format(W.shape, H.shape))

feature_names = vectorizer.get_feature_names()

n_top_words = 10

# %% print top words in each topic
for topic_idx, topic in enumerate(H):
    print("Topic #{}:".format(topic_idx))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

# %% create pandas dataframe for the topics
mydf = pd.DataFrame({"feature_name": feature_names})

for topic_idx, topic in enumerate(H):
    mydf["topic_{}".format(topic_idx)] = topic

mylist = list(mydf.itertuples())

mywords_topic1 = []
mywords_topic2 = []
mywords_topic3 = []
for order_id, key, num1, num2, num3 in mylist:
    mywords_topic1.append((key, num1))
    mywords_topic2.append((key, num2))
    mywords_topic3.append((key, num3))

mywords_topic1 = sorted(mywords_topic1, key=lambda myword: \
    myword[1], reverse=True)
mywords_topic2 = sorted(mywords_topic2, key=lambda myword: \
    myword[1], reverse=True)
mywords_topic3 = sorted(mywords_topic3, key=lambda myword: \
    myword[1], reverse=True)

% matplotlib
inline

from wordcloud import WordCloud
import matplotlib.pyplot as plt


def wdc(*mywords_topic):
    n_row = len(mywords_topic)
    n_col = 1
    plt.figure(figsize=(n_col * 3 * 1.618, n_row * 3))
    wordcloud = WordCloud()
    for index, item in enumerate(mywords_topic, start=1):
        wordcloud.fit_words(item)
        plt.subplot(n_row, n_col, index)
        plt.title('Topic {}'.format(index), size=16)
        plt.imshow(wordcloud)
        plt.axis("off")


wdc(mywords_topic1, mywords_topic2, mywords_topic3)