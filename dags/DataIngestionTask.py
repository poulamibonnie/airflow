import requests
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from OperatorConfig import TextModelConfig

class TextClassificationDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}

# implement base interface for following Builder pattern or abc class 
class DataIngestion(object):
    def __init__(self, config: TextModelConfig) -> None:
        self.config = config

    def build(self):
        pass 
    
    def run(self):
        response = requests.get(self.config.url, stream=True)

        with open('dataset.zip', 'wb') as file:
            file.write(response.content)

        with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        
        df = pd.read_csv('IMDB Dataset.csv')

        texts = df['review'].tolist()
        labels = [1 if sentiment == "positive" else 0 for sentiment in df['sentiment'].tolist()]

        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, self.config.max_length)
        val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, self.config.max_length)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)

        return tokenizer, train_dataset, val_dataset, train_dataloader, val_dataloader