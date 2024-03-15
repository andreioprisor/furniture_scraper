from crawler import Crawler
import sys
import pandas as pd
import time

try:
	if len(sys.argv) == 2:
		urls_file = sys.argv[1]
		urls = pd.read_csv(urls_file).iloc[:, 0].tolist()
		c = Crawler()
		start = time.time()
		c.crawl(urls, 0.5, threaded=False)
		print(f"Time taken: {time.time() - start}")
	else:
		raise Exception("No file provided")
except Exception as e:
	print(f"Exception occurred: {str(e)}")
