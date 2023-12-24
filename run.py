from crawler import Crawler
import pandas as pd

urls = pd.read_csv('furniture_links.csv').iloc[:, 0].tolist()
c = Crawler()

c.crawl(urls[:10], threaded=False)