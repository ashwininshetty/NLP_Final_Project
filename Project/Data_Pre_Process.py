#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np


# In[24]:


from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer


# In[ ]:


def pre_process_corpus(corpus):
    # Initialize a lemmatizer and create a set of English stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('English'))
    
    # Convert all text in the corpus to lowercase and split into words
    corpus = corpus.apply(lambda x: x.lower().split())
    
    # Filter out single-character words from each article
    corpus = corpus.apply(lambda article: [word for word in article if len(word) > 1])
    
    # Exclude words that are purely numeric
    corpus = corpus.apply(lambda article: [word for word in article if not word.isnumeric()])
    
    # Apply lemmatization to reduce words to their base or dictionary form
    corpus = corpus.apply(lambda article: [lemmatizer.lemmatize(word) for word in article])
    
    # Remove stop words to reduce noise in the text
    corpus = corpus.apply(lambda article: [word for word in article if word not in stop_words])
    
    # Eliminate mentions (words containing '@') from the text
    corpus = corpus.apply(lambda article: [word for word in article if '@' not in word])
    
    # Rejoin words into a single string per article after processing
    corpus = corpus.apply(lambda article: ' '.join(article))
    
    return corpus


# In[8]:


# Reading the train.csv
DF=pd.read_csv('train.csv', usecols=['label', 'tweet'])


# In[9]:


# Get first 10 rows from the dataframe
DF.head(10).values


# In[10]:


# Applying the function to pre-process the data on the tweet column
DF.tweet=pre_process_corpus(DF.tweet) 


# In[11]:


DF.tweet.head(10)


# In[12]:


# Getting data back to the CSV
DF.to_csv('my_dataset.csv', index=False)


# # Training and testing data split

# In[13]:


# Split the DataFrame 'DF' into a training set and a test set with a 70/30 ratio
df_train = DF[:round(0.7 * len(DF))]
df_test = DF[round(0.7 * len(DF)):]


# In[15]:


# Indices are reset to avoid issues during look-ups
df_test.reset_index(inplace=True, drop=True)


# # Embeddings, Bert and tfidf

# In[26]:


# Load the SentenceTransformer model with 'distilbert-base-nli-mean-tokens' for encoding sentences
bert_encoded = SentenceTransformer('distilbert-base-nli-mean-tokens')


# In[19]:


# Encode the 'tweet' column of the training DataFrame using the BERT model and save the encoded data as a numpy file
df_train_encoded = bert_encoded.encode(df_train.tweet)
np.save('encoded_train_tweet.npy', df_train_encoded)


# In[27]:


# Repeat same for test
df_test_enocded = bert_encoded.encode(df_test.tweet) 
np.save('encoded_test_tweet.npy', df_test_enocded)


# In[29]:


# Fit the TF-IDF vectorizer on the 'tweet' column of the training data and transform it into a dense matrix
tfidf = TfidfVectorizer()
tfidf_train_df_dense = tfidf.fit_transform(df_train.tweet).todense()

# Transform the 'tweet' column of the test data using the fitted TF-IDF vectorizer into a dense matrix
tfidf_test_df_dense = tfidf.transform(df_test.tweet).todense()


# In[30]:


# Save the dense TF-IDF matrices of the training and test tweets to numpy files
np.save('tfidf_train_tweet.npy', tfidf_train_df_dense)
np.save('tfidf_test_tweet.npy', tfidf_test_df_dense)


# In[ ]:




