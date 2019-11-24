import query_processor

def search_news():
    query = input("Search for: ")
    date_filter = input("Do you want to filter results by date? For yes press ""Y"" for no press Enter ")
    if date_filter is "Y":
        dt = input("Please enter date in format DD/MM/YYYY. \nSingle date, comma separated date interval (DD/MM/YYYY - DD/MM/YYYY) or ""today"" is accepted. ")
    else:
        dt = None
    query_processor.search(query, dt)

search_news()