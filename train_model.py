from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
import pyarrow as pa
from datasets import Dataset
import pandas as pd
import torch
from transformers import TrainingArguments, Trainer
import torch
from transformers import TrainingArguments, Trainer
import glob
import pandas as pd
import pyarrow as pa
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class TrainModel():
	def __init__(self, tokenizer, pretrained, num_labels):
		self.tokenizer = tokenizer
		self.pretrained = pretrained
		self.num_labels = num_labels
            
	def build_df(self, directory_path):
		txt_files = glob.glob(f'{directory_path}/*.txt')
		# # Create a list to hold the dataframes
		dfs = []

		# Read each .txt file and append it to the list of dataframes
		for txt_file in txt_files:
			df = pd.read_csv(txt_file, sep='\t', names=["text", "label"], 
							encoding='utf-8', on_bad_lines='skip', quoting=3, escapechar="\\")
			dfs.append(df)

		# # Concatenate all dataframes in the list
		combined_df = pd.concat(dfs, ignore_index=True).drop_duplicates().reset_index(drop=True)
		return combined_df
	
	# Process the data by tokenizing the text and convertin the labels to integers
	def process_data(self, row):
		text = row['text']
		text = str(text)
		text = ' '.join(text.split())
		encodings = self.tokenizer(text,truncation=True)
		label = 0
		if row['label'] == 1:
			label += 1
		encodings['label'] = label
		encodings['text'] = text
		return encodings
	
	# Prepare the data for training by shuffling and splitting into training and validation sets
	def prepare_data(self, df):
		processed_data = [self.process_data(df.iloc[i]) for i in range(1,df.shape[0])]
		new_df = pd.DataFrame(processed_data)
		train_df, valid_df = train_test_split(
			new_df,
			test_size=0.19,
			random_state=2022
		)
		train_hg = Dataset(pa.Table.from_pandas(train_df)).shuffle(seed=2022)
		valid_hg = Dataset(pa.Table.from_pandas(valid_df)).shuffle(seed=2022)
		return train_hg, valid_hg
	
	def compute_metrics(self, pred):
		labels = pred.label_ids
		preds = pred.predictions.argmax(-1)
		precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
		acc = accuracy_score(labels, preds)
		return {
			'accuracy': acc,
			'f1': f1,
			'precision': precision,
			'recall': recall
		}

	def train_model(self, train_hg, valid_hg, output_dir, output_logs):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.pretrained.to(device)
		
		for param in self.pretrained.base_model.parameters():
			param.requires_grad = False

		for param in self.pretrained.classifier.parameters():
			param.requires_grad = True
		
		training_args = TrainingArguments(
			output_dir=output_dir,
			num_train_epochs=20,
			per_device_train_batch_size=16,
			per_device_eval_batch_size=64,
			warmup_steps=1000,
			weight_decay=0.02,
			logging_dir=output_logs ,  # Specify GPU device
		)
		trainer = Trainer(
			model=self.pretrained,
			args=training_args,
			train_dataset=train_hg,
			eval_dataset=valid_hg,
			compute_metrics=self.compute_metrics,
			tokenizer=self.tokenizer
		)
		trainer.train()
		return trainer

model = AutoModelForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2
)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


tm = TrainModel(tokenizer, model, num_labels=2)

df = tm.build_df('training_data')

data = tm.prepare_data(df)
trainer = tm.train_model(data[0], data[1], 'model_results', 'logs')
trainer.save_model('checkpoint')