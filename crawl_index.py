#import functions from other scripts
import crawler
import indexer

#flag - is this first time run?
is_first = False

#variables for running crawler
OUTPUT_DIR = "data/"
BBC_world = "http://feeds.bbci.co.uk/news/world/rss.xml"
BBC_uk = "http://feeds.bbci.co.uk/news/uk/rss.xml"

#variables for updating index
INDEX_PATH = OUTPUT_DIR + "index.json"
VECTOR_PATH = OUTPUT_DIR + "doc_vecs.csv"

#run the crawler
perc_world, added_world = crawler.crawl(URL = BBC_world, PATH = OUTPUT_DIR)
perc_uk, added_uk = crawler.crawl(URL = BBC_uk, PATH = OUTPUT_DIR)

#add to index and compute average word2vec
indexer.update_index_vecs(df = added_world, index_path = INDEX_PATH, vec_path = VECTOR_PATH, first_time=is_first)
indexer.update_index_vecs(df = added_uk, index_path = INDEX_PATH, vec_path = VECTOR_PATH, first_time=is_first)
