import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from abc import ABC, abstractmethod
from ModelBuildingTask import ModelBuildFactory, TextModelConfig, BertClassifer, LSTMRNN

class AbstractModelTrainer(ABC):
    @abstractmethod
    def train(self, model, data_source, hyperparameters):
        pass

class ModelTrainingFactory(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_trainer(config: TextModelConfig, **kwargs) -> AbstractModelTrainer:
        if config.model_type == "BERT":
            return BERTTrainer(**kwargs)
        elif config.model_type == "LSTM":
            return LSTMTrainer(**kwargs)
        else:
            raise NotImplementedError("Error: The required model is not supported by the Text Operator.")

class BERTTrainer(AbstractModelTrainer):
    def __init__(self, model, data_source, hyperparameters):
        self.model = model
        self.data_source = data_source
        self.hyperparameters = hyperparameters

    def train(self, model, data_source, hyperparameters):
        # Load the data
        train_dataloader = data_source['train']

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(hyperparameters['epochs']):
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                optimizer.zero_grad()
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch + 1} Train Loss: {avg_train_loss:.3f}')

        return model

class LSTMTrainer(AbstractModelTrainer):
    def __init__(self, model, data_source, hyperparameters):
        self.model = model
        self.data_source = data_source
        self.hyperparameters = hyperparameters

    def train(self, model, data_source, hyperparameters):
        # Load the data
        train_dataloader = data_source['train']

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(hyperparameters['epochs']):
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_data = batch['input_data']
                labels = batch['labels']

                optimizer.zero_grad()
                logits = model(input_data)
                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch + 1} Train Loss: {avg_train_loss:.3f}')

        return model

class ModelTraining(object):
    def __init__(self, config) -> None:
        self.config = config

    def train(self, model, data_source, hyperparameters):
        trainer = ModelTrainingFactory.create_trainer(self.config, model=model, data_source=data_source, hyperparameters=hyperparameters)
        return trainer.train(model, data_source, hyperparameters)