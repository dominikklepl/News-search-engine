#import packages
import pandas as pd
import feedparser

def crawl_RSS(URL, PATH):
    # load RSS feed from specified URL
    feed = feedparser.parse(URL)

    feed_len = len(feed.entries) #number of news in feed
    old_news = 0  # count how many news in feed were already scraped

    #load database of previously stored news
    meta_data = pd.read_csv(PATH + "database.csv", index_col = 'Unnamed: 0')

    #given one RSS entry return list of ID, title, summary and link
    def process_entry(entry, ID):
        ID = ID
        title = entry.title
        link = entry.link
        published = str(entry.published_parsed.tm_mday) + '/' + str(entry.published_parsed.tm_mon) + '/' + str(entry.published_parsed.tm_year)
        return [ID, title, link, published]

    #loop through all entries, extract info and save in a pandas dataframe
    data = [] #dataframe for saving the extracted info
    n = len(meta_data)+1 #ID value
    for i in range(len(feed.entries)):
        entry = feed.entries[i]
        #check that link isn't in the database yet
        if entry.link not in meta_data['link'].values:
            processed = process_entry(entry = entry, ID=n)
            data.append(processed)
            n += 1
        else: old_news += 1

    #if some new entries were present, correct the IDs and add it to the database
    if len(data) > 0:
        #transform data to pandas DataFrame
        news_extracted = pd.DataFrame(data, columns=['ID', 'title', 'link', 'published'])

        #correct the IDs so that there are no duplicates in the database
        news_extracted['ID'] = news_extracted['ID']+(len(meta_data)) + 1

        #add new news to the database
        meta_data = pd.concat([meta_data, news_extracted], axis = 0)

        #write database to a csv file
        meta_data.to_csv(PATH + "database.csv")

    else: print("No new entries found")

    already_scraped = (old_news/feed_len)*100
    print("{} % of entries were already scraped.".format(already_scraped))

    return already_scraped

#variables for running crawler
OUTPUT_DIR = "data/"
BBC_URL = "http://feeds.bbci.co.uk/news/world/rss.xml"

#run the crawler
new = crawl_RSS(URL = BBC_URL, PATH = OUTPUT_DIR)