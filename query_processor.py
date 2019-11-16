#import
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import string
import json
import gensim
import numpy as np
import pandas as pd


#tag words
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

#remove stopwords and lemmatize
stop = stopwords.words('english')
lem = WordNetLemmatizer()
def stop_lemmatize(doc):
    tokens = nltk.word_tokenize(doc)
    tmp = ""
    for w in tokens:
        if w not in stop:
            tmp += lem.lemmatize(w, get_wordnet_pos(w)) + " "
    return tmp

def process_string(text):
  text = text.lower() #to lowercase
  text = text.translate(str.maketrans('', '', string.punctuation)) #strip punctuation
  text = stop_lemmatize(text)
  return text

word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
def average_vectors(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
    if len(doc) == 0:
      return np.zeros(300)
    else:
      return np.mean(word2vec_model[doc], axis=0)

def process_query(query):
  norm = process_string(query)
  return norm.split()

def lists_intersection(lists):
  intersect = list(set.intersection(*map(set, lists)))
  intersect.sort()
  return intersect

def connect_id_df(retrieved_id, df):
    return df[df.ID.isin(retrieved_id)].reset_index(drop=True)

with open('data/index.json', 'r') as f:
    index = json.load(f)
database = pd.read_csv("data/database.csv", index_col = 'Unnamed: 0')
doc_vecs = pd.read_csv("data/doc_vecs.csv", index_col= 'Unnamed: 0')

def search_googleish(query, index=index):
  query_split = process_query(query)
  retrieved = []
  for word in query_split:
    if word in index.keys():
      retrieved.append(index[word])
  if len(retrieved)>0:
    result = lists_intersection(retrieved)
  else:
      result = [0]
  result = connect_id_df(result, database)
  return result

def cos_similarity(a, b):
  dot = np.dot(a, b)
  norma = np.linalg.norm(a)
  normb = np.linalg.norm(b)
  cos = dot / (norma * normb)
  return(cos)

def rank_results(query, results):
  query_norm = process_query(query)
  query_vec = average_vectors(word2vec, query_norm)
  result_vecs = connect_id_df(results.ID, doc_vecs)
  cos_sim = []
  for i in range(len(result_vecs)):
    doc_vec = result_vecs.loc[i,:].drop(['ID'])
    cos_sim.append(cos_similarity(doc_vec, query_vec))
  results['rank'] = cos_sim
  results = results.sort_values('rank', axis=0)
  return results

def print_results(result_df):
  for i in range(len(result_df)):
    res = result_df.loc[i, :]
    print(res.title)
    print(res.summary)
    if i == len(result_df):
        print(res.link)
    else:
        print("{}\n" .format(res.link))

def search(query):
    result = search_googleish(query)
    result = rank_results(query, result)
    print_results(result)



