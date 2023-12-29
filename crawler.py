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

options = Options()
service = Service(ChromeDriverManager().install())

class Crawler:
	def __init__(self):
		self.lock = None
		self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model = AutoModelForSequenceClassification.from_pretrained('./model').to(device)

	def get_prediction(self, tokenizer, model, text):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		encoding = tokenizer(text, return_tensors='pt', truncation=True)
		encoding = {k: v.to(device) for k, v in encoding.items()}

		outputs = model(**encoding)

		logits = outputs.logits
		probs = torch.nn.functional.softmax(logits, dim=-1)
		sigmoid = torch.nn.Sigmoid()
		probs = sigmoid(logits.squeeze().detach())
		probs = probs.detach().numpy()
		if np.argmax(probs, axis=-1) == 1:
			return text

	def scraper(self, urls):
		products = []
		driver = webdriver.Chrome(service=service, options=options)
		driver.set_page_load_timeout(10)
		driver.set_script_timeout(15)
		for url in urls:
			print(f"Crawling {url}...")
			products = []
			try:
				driver.get(url)
				# Get the page source
				products = []
				page_source = driver.page_source
				soup = bs(page_source, 'html.parser')
				links = soup.find_all('a')
				headers = soup.find_all(['h1','h2','h3','h4','h5','h6', 'p'])

				for header in headers:
					if self.get_prediction(self.tokenizer, self.model, header.get_text(strip=True)) is not None:
						products.append(header.get_text(strip=True))

				for link in links:
					text = link.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
					if text is None:
						texts = link.get_text(separator=' | ',strip=True).split(' | ')
						for text in texts:
							if self.get_prediction(self.tokenizer, self.model, text) is not None:
								products.append(text)
								break
			except Exception as e:
				print(f"Exception occurred while crawling: {str(e)}")
				continue
			
			if products and self.lock is not None:
				with self.lock:
					with open('products.txt', 'a') as f:
						f.write(urlparse(url).netloc + '\n')
						for product in products:
							f.write(product + '\n')
			elif products:
				with open('products.txt', 'a') as f:
					f.write(urlparse(url).netloc + '\n')
					for product in products:
						f.write(product + '\n')
		driver.quit()


	def crawl(self, urls, threaded, num_workers = None):
		if threaded:
			threads = []
			self.lock = threading.Lock()
			for _ in range(num_workers):
				t = threading.Thread(target=self.scraper, args=(urls,))
				t.start()
				threads.append(t)
			
			for thread in threads:
				thread.join()
		else:
			self.scraper(urls)