import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from ModelBuildingTask import (
    ModelBuildFactory,
    TextModelConfig,
    BertClassifer,
    LSTMRNN,
    AbstractModelsBuilder
)
from transformers import get_linear_schedule_with_warmup
from AirflowDeepLearningOperatorLogger import OperatorLogger

logger = OperatorLogger.getLogger()


class AbstractModelTrainer(ABC):
    @abstractmethod
    def train(self, model, train_dataloader):
        raise NotImplementedError()


class ModelTrainingFactory(object):
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_trainer(config: TextModelConfig) -> AbstractModelTrainer:
        if config.model_type == "BERT":
            return BERTTrainer(config)
        elif config.model_type == "LSTM":
            return LSTMTrainer(config)
        else:
            raise NotImplementedError("Error: The required model is not supported by the Text Operator.")


class BERTTrainer(AbstractModelTrainer):
    def __init__(self, config: TextModelConfig):
        self.config = config 

    def train(self, model: object, train_dataloader: object):
        # Load the data

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        loss_fn = nn.CrossEntropyLoss()
        total_steps = len(train_dataloader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        logger.info("[x] Init the training for BERT")
        logger.info("[x] Total number of epochs ")
        print(self.config.num_epochs)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                print('batch :', batch) 
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.config.device)
                attention_mask = batch['attention_mask'].to(self.config.device)
                labels = batch['label'].to(self.config.device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)
            logger.info(f'Epoch {epoch + 1} Train Loss: {avg_train_loss:.3f}')

        return model


class LSTMTrainer(AbstractModelTrainer):
    def __init__(self, model: LSTMRNN, data_source, hyperparameters):
        self.model = model
        self.data_source = data_source
        self.hyperparameters = hyperparameters

    def train(self, model: LSTMRNN, data_source, hyperparameters):
        # Load the data
        train_dataloader = data_source['train_dataloader']

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(hyperparameters['num_epochs']):
            print(f"Epoch {epoch + 1}/{hyperparameters['num_epochs']}")
            model.train()
            for batch in train_dataloader:
                text = batch['input_ids'].to(model.device)
                labels = batch['label'].to(model.device)

                optimizer.zero_grad()
                model.extra_args['text'] = text  # Set the text attribute for LSTM model
                logits = model()
                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_train_loss = total_loss / len(train_dataloader)
            logger.info(f'Epoch {epoch + 1} Train Loss: {avg_train_loss:.3f}')

        return model


class ModelTraining(object):
    def __init__(self, config: TextModelConfig) -> None:
        self.config = config

    def train(self, model: AbstractModelsBuilder, data_loader: object):
        trainer = ModelTrainingFactory.create_trainer(
            self.config
        )
        return trainer.train(model, data_loader) 

    def run(self, model: AbstractModelsBuilder, data_loader: object):
        logger.info("[x] Started the training")
        model = self.train(model=model, data_loader=data_loader)
        logger.info("[x] Model training ended")
        return model