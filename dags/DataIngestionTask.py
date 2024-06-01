import requests
import zipfile
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

from OperatorConfig import TextModelConfig
from AirflowDeepLearningOperatorLogger import OperatorLogger
import os

logger = OperatorLogger.getLogger()
# A custom dataset class for text classification
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

        # Tokenize text using the tokenizer and convert it into PyTorch tensors
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)

        return {
            'input_ids': encoding['input_ids'].flatten(), 
            'attention_mask': encoding['attention_mask'].flatten(), 
            'label': torch.tensor(label),
        }

class DataIngestion(object):
    def __init__(self, config: TextModelConfig) -> None:
        self.config = config

    def build(self):pass 
    
    
    def run(self):
        logger.info("Starting data ingestion process.")
        # Download dataset from the provided URL
        logger.info(f"Downloading dataset from {self.config.url}.")
        response = requests.get(self.config.url, stream=True)
        with open('dataset.zip', 'wb') as file:
            file.write(response.content)
        logger.info("Dataset downloaded successfully.")
        
        # Extract dataset from the downloaded zip file
        logger.info("Extracting dataset from the zip file.")
        with zipfile.ZipFile('dataset.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        logger.info("Dataset extracted successfully.")
        
        
        # Read dataset into a pandas DataFrame
        df = pd.read_csv('IMDB Dataset.csv')

        # Extract texts and labels from the DataFrame
        texts = df['review'].tolist()
        labels = [1 if sentiment == "positive" else 0 for sentiment in df['sentiment'].tolist()]

        # Split dataset into train and validation sets
        train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.5, random_state=42)

        # Initialize tokenizer using the provided BERT model name
        logger.info(f"Initializing tokenizer with model name: {self.config.bert_model_name}.")
        tokenizer = DistilBertTokenizer.from_pretrained(self.config.bert_model_name)
        
        # Create train and validation datasets using the custom dataset class
        logger.info("Creating train and validation datasets.")
        train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, self.config.max_length)
        test_dataset = TextClassificationDataset(test_texts, test_labels, tokenizer, self.config.max_length)
        

        # Create train and validation dataloaders
        logger.info("Creating train and validation dataloaders.")
        train_dataloader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        logger.info("Data ingestion process completed successfully.")

        return tokenizer, train_dataset, test_dataset, train_dataloader, test_dataloader