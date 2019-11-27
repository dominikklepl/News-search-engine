#import
import crawler
import indexer
import time
import numpy as np

#flag - is this first time run?
is_first = False

#where to save database and index
OUTPUT_DIR = "data/"

#read list of feeds from txt file
with open('data/feed_list.txt', 'r') as f:
    URLS = f.read().split(",")

#variables for updating index
INDEX_PATH = OUTPUT_DIR + "index.json"
VECTOR_PATH = OUTPUT_DIR + "doc_vecs.csv"

#iterate over feeds in the URLS list
import os
vecs_size = round(os.path.getsize("data/doc_vecs.csv")/1000000, 2)
n = 6000
while vecs_size < 1500:
    new_stuff = []
    for url in URLS:
        print("Crawling {}" .format(url))
        # run the crawler
        perc, added = crawler.crawl(URL=url, PATH=OUTPUT_DIR)
        new_stuff.append(perc)
        # add to index and compute average word2vec
        indexer.update_index_vecs(df=added, index_path=INDEX_PATH, vec_path=VECTOR_PATH, first_time=is_first)

    #update site of vecs file
    vecs_size = round(os.path.getsize("data/doc_vecs.csv") / 1000000, 2)
    print(vecs_size)

    mean = np.mean(new_stuff)
    if mean >= 80:
        n += 3600
        print("Increasing hit rate by 1 hour.")
    if mean < 80:
        n -= 3600
        print("Decreasing hit rate by 1 hour.")

    time.sleep(n)


