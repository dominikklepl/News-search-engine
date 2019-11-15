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

def build_index(df, index_path="index.json", first_time=False):
    if df is None:
        print("Nothing to update")
    else:
        if first_time:
            index = {}
        else:
            with open(index_path, 'r') as old_idx:
                index = json.load(old_idx)

        to_add = transform_df(df)
        index = index_all(df = to_add, index = index)

        #write to json
        with open(index_path, 'w') as new_idx:
            json.dump(index, new_idx, sort_keys=True, indent=4)




