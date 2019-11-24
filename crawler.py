#import packages
import pandas as pd
import feedparser


# given entry return list of ID, title, summary, link and date published
def process_entry(entry, ID_n):
  ID = ID_n
  title = entry.title
  summary = entry.summary
  link = entry.link
  published = str(entry.published_parsed.tm_mday) + '/' + \
              str(entry.published_parsed.tm_mon) + '/' + \
              str(entry.published_parsed.tm_year)
  return [ID, title, summary, link, published]

def crawl(URL, PATH = "data/"):
    # load RSS feed from specified URL
    feed = feedparser.parse(URL)

    feed_len = len(feed.entries) #number of news in feed
    old_news = 0  # count how many news in feed were already scraped

    #load database of previously stored news
    meta_data = pd.read_csv(PATH + "database.csv", index_col = 'Unnamed: 0')

    #loop through all entries, extract info and save in a pandas dataframe
    data = [] #dataframe for saving the extracted info

    meta_len = len(meta_data)
    n = meta_len + 1
    for i in range(feed_len):
        entry = feed.entries[i]
        #check that link isn't in the database yet
        if entry.link not in meta_data['link'].values:
            processed = process_entry(entry = entry, ID_n = n)
            data.append(processed)
            n = n + 1
        else: old_news += 1

    already_scraped = (old_news/feed_len)*100
    print("{} % of entries were already scraped.".format(already_scraped))

    # save info about scraped:new ratio
    new_scraped = {"ratio": [already_scraped]}
    new_scraped = pd.DataFrame.from_dict(new_scraped)
    new_scraped.to_csv(PATH + "rate.csv", index=False)

    #if some new entries were present, correct the IDs and add it to the database
    if len(data) > 0:
        #transform data to pandas DataFrame
        news_extracted = pd.DataFrame(data, columns=['ID', 'title', 'summary', 'link', 'published'])

        #add new news to the database
        meta_data = pd.concat([meta_data, news_extracted], axis = 0)

        #write database to a csv file
        meta_data.to_csv(PATH + "database.csv")

        return [already_scraped, news_extracted]

    else:
        print("No new entries found")
        return [already_scraped, None]