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

#run only first time
#wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

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

def transform_df(df):
  df['title'] = df['title'].apply(process_string)
  df['summary'] = df['summary'].apply(process_string)
  df['text'] = df['title'] + " " + df['summary']
  drop_cols = ['title', 'summary', 'published', 'link']
  df = df.drop(drop_cols, axis=1)
  return df

def index_it(entry, index):
  words = entry.text.split()
  ID = int(entry.ID)
  for word in words:
    if word in index.keys():
      if ID not in index[word]:
        index[word].append(ID)
    else:
      index[word] = [ID]
  return index

def index_all(df, index):
  for i in range(len(df)):
    entry = df.loc[i,:]
    index = index_it(entry = entry, index = index)
  return index

def build_index(transformed_df, index_path="index.json", first_time=False):
    if first_time:
        index = {}
    else:
        with open(index_path, 'r') as old_idx:
            index = json.load(old_idx)

    index = index_all(df=transformed_df, index=index)

    # write to json
    with open(index_path, 'w') as new_idx:
        json.dump(index, new_idx, sort_keys=True, indent=4)

word2vec = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
def average_vectors(word2vec_model, doc):
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.vocab]
    if len(doc) == 0:
      return np.zeros(300)
    else:
      return np.mean(word2vec_model[doc], axis=0)

def get_vectors_all(transformed_df, vec_path="doc_vecs.csv", first_time=False):
    if first_time:
        doc_vecs = {}
    else:
        doc_vecs = pd.read_csv(vec_path)

    for i in range(len(transformed_df)):
        row = transformed_df.loc[i, :]
        text = row.text.split()
        ID = int(row.ID)
        doc_vecs[str(ID)] = average_vectors(word2vec, text)

    #to pandas dataframe
    doc_vecs = pd.DataFrame.from_dict(data=doc_vecs, orient="index")
    doc_vecs['ID'] = doc_vecs.index

    # write to csv
    doc_vecs.to_csv((vec_path))


def update_index_vecs (df, index_path="index.json", vec_path="doc_vecs.csv", first_time=False):
    if df is None:
        print("Nothing to update")
    else:
        to_add = transform_df(df)
        build_index(transformed_df=to_add, index_path=index_path, first_time=first_time)
        get_vectors_all(transformed_df=to_add, vec_path=vec_path, first_time=first_time)





