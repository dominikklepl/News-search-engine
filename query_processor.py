#import
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import string
import json
import gensim
import numpy as np
import pandas as pd
from datetime import date, timedelta
import time
from contextlib import redirect_stdout
import webbrowser
from urllib.request import pathname2url
import os
from bs4 import BeautifulSoup

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
  result_df = result_df.reset_index(drop = True)
  for i in range(len(result_df)):
    res = result_df.loc[i, :]
    print(res.title)
    print(res.summary)
    print(res.published)
    if i == len(result_df):
        print(res.link)
    else:
        print("{}" .format(res.link))
        print("---------")

#date restriction
def get_today():
  today = date.today()
  today = today.strftime("%d/%m/%Y")
  today = [today]
  return today

def daterange(start, end):
    for n in range(int ((end - start).days)+1):
        yield start + timedelta(n)

def format_date(dt):
  dt = dt.split("/")
  dt = date(int(dt[2]), int(dt[1]), int(dt[0]))
  return dt

def date_interval(interval):
  interval = interval.split("-")
  start = format_date(interval[0])
  end = format_date(interval[1])
  interval = []
  for dt in daterange(start, end):
      interval.append(dt.strftime("%d/%m/%Y"))
  return interval

def filter_date(dat, df):
    if len(df) == 0:
        return df
    else:
        if "today" in dat:
            dat = get_today()
        if len(dat) == 10:
            dat = [dat]
        if len(dat) > 11:
            dat = date_interval(dat)

        result = df[df.published.isin(dat)].reset_index(drop=True)
        return result

def print_html_long(txt_output):
    soup = BeautifulSoup(features="html.parser")
    style = soup.new_tag("style")
    style.string = ".content {\nmax-width: 600px;\nmargin: auto;\n}"
    soup.insert(0, style)
    body = soup.new_tag('body')
    soup.insert(1, body)
    div = soup.new_tag("div")
    div["class"] = "content"
    body.insert(0, div)
    table = soup.new_tag('table')
    # table["style"] = "width:80%"
    div.insert(0, table)

    f = open(txt_output, 'r')
    docs = f.read().split("---------")

    line = docs[0].split("\n")[0]
    row = soup.new_tag('tr')
    col = soup.new_tag('th')
    col.string = line
    row.insert(0, col)
    table.insert(len(table.contents), row)

    docs[0] = docs[0].split("seconds)")[1]

    for line in docs:
        texts = line.split("\n")
        for text in texts:
            row = soup.new_tag('tr')
            col = soup.new_tag('td')

            if "www" in text:
                link = soup.new_tag("a", href=text)
                link.string = text
                col.insert(0, link)
            else:
                col.string = text
            row.insert(0, col)
            table.insert(len(table.contents), row)

    with open('data/output.html', 'w') as outfile:
        outfile.write(soup.prettify())

    url = 'file:{}'.format(pathname2url(os.path.abspath('data/output.html')))
    webbrowser.open_new_tab(url)

def print_html_short():
    soup = BeautifulSoup(features="html.parser")
    style = soup.new_tag("style")
    style.string = ".content {\nmax-width: 600px;\nmargin: auto;\n}"
    soup.insert(0, style)
    body = soup.new_tag('body')
    soup.insert(1, body)
    div = soup.new_tag("div")
    div["class"] = "content"
    body.insert(0, div)
    table = soup.new_tag('table')
    div.insert(0, table)
    row = soup.new_tag('tr')
    col = soup.new_tag('th')
    col.string = "No articles matching search criteria found."
    row.insert(0, col)
    table.insert(len(table.contents), row)

    with open('data/output.html', 'w') as outfile:
        outfile.write(soup.prettify())

    url = 'file:{}'.format(pathname2url(os.path.abspath('data/output.html')))
    webbrowser.open_new_tab(url)


#search engine itself
def search(query, dat=None):
    start = time.time()

    result = search_googleish(query)
    result = rank_results(query, result)

    if dat is not None:
        result = filter_date(dat, result)

    n_retrieved = len(result)
    end = time.time()
    time_taken = round((end - start), 2)

    if n_retrieved == 0:
        print("No articles matching search criteria found.")
        print_html_short()
    else:
        print("({} results found in {} seconds)\n".format(n_retrieved, time_taken))
        print_results(result)

        with open('data/output.txt', 'w') as f:
            with redirect_stdout(f):
                print("({} results found in {} seconds)".format(n_retrieved, time_taken))
                print_results(result)
        print_html_long('data/output.txt')

def search_news():
    query = str(input("Search for: "))
    filter = str(input("Do you want to filter results by date? For yes press ""Y"" for no press Enter "))
    if filter is "Y":
        dt = str(input("Please enter date in format DD/MM/YYYY. Single date, comma separated date interval (DD/MM/YYYY - DD/MM/YYYY) or ""today"" is accepted. "))
    else:
        dt = None
    print(type(dt))

    search(query=query, dat=dt)