Implementation of the Azure Notebooks Python Wordcloud from Tweets file example


1 Introduction
Sentiment analysis is commonly used in marketing and customer service to answer questions such as "Is a product review positive or negative?" and "How are customers responding to a product release?" etc.

Topic modeling discovers the abstract "topics" in a corpus of texts. The results from topic modeling analysis can be used in sentiment analysis. For example, they can be used to split texts into different subsets and allow us to train a separate sentiment model for each of the subsets. Training separate sentiment models for different subsets can lead to more accurate predictions than using a single model for all the texts.

The purpose of this notebook is to illustrate how to discover and visualize topics from a corpus of Twitter tweets using Jupyter notebook.

2 Data
2.1 Data Source
The dataset used in his example is based on the Sentiment140 dataset. The Sentiment140 dataset has approximately 1,600,000 automatically annotated tweets and 6 fields for each tweet. For illustration purpose, a sample of the Sentiment140 dataset will be used. This sample has 160,000 tweets and two fields for each tweet - the polarity of the tweet and the text of the tweet. The sample dataset is located here.

Download this dataset by running the following command

!curl -L -o mydatafile.csv http://azuremlsamples.azureml.net/templatedata/Text%20-%20Input.csv
After downloading the dataset, upload it to OneDrive or Dropbox as "mydatafile.csv" and import it into this notebook using the "Data" menu.