# -*- coding: utf-8 -*-

#import
import numpy as np
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeClassifier
from joblib import dump, load
import spacy
from html import unescape

# create a spaCy tokenizer
spacy.load('en')
lemmatizer = spacy.lang.en.English()

#to lower, remove HTML tags
def my_preprocessor(doc):
    return(unescape(doc).lower())

# tokenize the doc and lemmatize its tokens
def my_tokenizer(doc):
    tokens = lemmatizer(doc)
    return([token.lemma_ for token in tokens])

#load trained model
model = load("data/news_classifier.joblib")

def predict_topic():
  text = input("Enter text to be predicted: ")
  prediction = model.predict([text])[0]
  print("This text is most likely about {}." .format(prediction))

while True:
    predict_topic()