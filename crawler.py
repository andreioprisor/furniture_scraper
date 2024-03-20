from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup as bs
import torch
import numpy as np
import threading
from urllib.parse import urlparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import time
import sys
from selenium.webdriver.common.by import By

options = Options()
options.add_argument('--headless')
service = Service(ChromeDriverManager().install())

class Crawler:
	def __init__(self):
		self.lock = None
		self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = AutoModelForSequenceClassification.from_pretrained('checkpoint-76000').to(device)

	def get_prediction(self, tokenizer, model, text):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=16)
		encoding = {k: v.to(device) for k, v in encoding.items()}
		outputs = model(**encoding)
		logits = outputs.logits
		probs = torch.nn.functional.softmax(logits, dim=-1)
		return probs[0][1].item()

	def bypass_popup(self, driver):
		xpath_expressions = []
		xpath_expressions.extend([
			'//button[contains(translate(text(),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"), "accept")]',
			'//button[contains(translate(text(),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"), "agree")]',
			'//button[contains(translate(text(),"ABCDEFGHIJKLMNOPQRSTUVWXYZ","abcdefghijklmnopqrstuvwxyz"), "close")]',
		])
		for xpath in xpath_expressions:
			try:
				a = driver.find_element(By.XPATH, xpath)
				a.click()
				break
			except Exception as e:
				pass
		js_script = """
		var elements = document.querySelectorAll('a', 'button');
		for (var i = 0; i < elements.length; i++) {
			for (var j = 0; j < elements[i].attributes.length; j++) {
				var attribute = elements[i].attributes[j];
				if (attribute.value.indexOf('accept') > -1 || attribute.value.indexOf('agree') > -1 || attribute.value.indexOf('close') > -1){
					elements[i].click(); // Perform the click action on the matching element
					return true; // Return true to indicate an element was clicked
				}
			}
		}
		return false; // Return false if no matching element was found to click
		"""
		try:
			driver.execute_script(js_script)
		except Exception as e:
			pass
	
	## This function is used to scrape the products from the given urls by returning top candidates based on the probability
	def scraper(self, urls, threshold, lock=None):
		products = []
		options = Options()
		options.add_argument('--headless')
		service = Service(ChromeDriverManager().install())
		driver = webdriver.Chrome(service=service, options=options)
		driver.set_page_load_timeout(10)
		driver.set_script_timeout(15)
		for url in urls:
			print(f"Scraping {url}...")
			products = []
			try:
				driver.get(url)
				self.bypass_popup(driver)
				# Get the page source
				products = []
				page_source = driver.page_source
				soup = bs(page_source, 'html.parser')
				links = soup.find_all('a')
				headers = soup.find_all(['h1','h2','h3','h4','h5','h6', 'p', 'span', 'font'])

				for header in headers:
					t = header.get_text(separator='~~',strip=True)
					products.append((t, self.get_prediction(self.tokenizer, self.model, t)))
				# Many websites have their products in the form of links, so we need to extract the text from those links to infer in the model
				for link in links:
					t = link.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span', 'font'])
					if t is None:
						texts = link.get_text(separator='~~',strip=True).split('~~')
						for text in texts:
							products.append((text, self.get_prediction(self.tokenizer, self.model, text)))
				products = sorted(products, key=lambda x: x[1], reverse=True)
				products = [p for p in products if p[1] > threshold]
			except Exception as e:
				print(f"Exception occurred while crawling: {str(e)}")
				continue
			if lock:
				lock.acquire()
				with open('products_threaded.csv', 'a') as f:
					f.write(f'''{url}, "{str(products).replace('"', '')}"\n''')
				lock.release()
			else:
				with open('products1.csv', 'a') as f:
					f.write(f'''{url}, "{str(products).replace('"', '')}"\n''')
		driver.quit()

	# Function used to crawl the given urls either in a multi-threaded or single-threaded manner
	def crawl(self, urls, threshold, threaded, num_workers = None):
		if threaded:
			threads = []
			lock = threading.Lock()
			for i in range(num_workers):
				start = i * len(urls) // num_workers
				end = (i + 1) * len(urls) // num_workers
				if i == num_workers - 1:
					end = len(urls)
				segment = urls[start:end]
				t = threading.Thread(target=self.scraper, args=(segment, threshold, lock))
				t.start()
				threads.append(t)
			
			for thread in threads:
				thread.join()
		else:
			self.scraper(urls, threshold)

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
