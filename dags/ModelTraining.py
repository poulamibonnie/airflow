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
from AirflowDeepLearningOperatorLogger import OperatorLogger

logger = OperatorLogger.getLogger()


class AbstractModelTrainer(ABC):
    @abstractmethod
    def train(self, model, data_source, hyperparameters):
        raise NotImplementedError()


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
    def __init__(self, model: BertClassifer, data_source, hyperparameters):
        self.model = model
        self.data_source = data_source
        self.hyperparameters = hyperparameters

    def train(self, model: BertClassifer, data_source, hyperparameters):
        # Load the data
        train_dataloader = data_source['train_dataloader']

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(hyperparameters['num_epochs']):
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch['attention_mask'].to(model.device)
                labels = batch['label'].to(model.device)

                optimizer.zero_grad()
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

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
            model.train()
            total_loss = 0
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
    def __init__(self, config: TextModelConfig, model: AbstractModelsBuilder, data_source: dict) -> None:
        self.config = config
        self.model = model
        self.data_source = data_source

        # Set the device (GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        hyperparameters = {
            'learning_rate': self.config.optimizer_learn_rate,
            'num_epochs': self.config.num_epochs,
            'batch_size': self.config.batch_size
        }

        trainer = ModelTrainingFactory.create_trainer(
            self.config,
            model=self.model,
            data_source=self.data_source,
            hyperparameters=hyperparameters
        )
        return trainer.train(self.model, self.data_source, hyperparameters)

    def run(self):
        logger.info("[x] Started the training")
        self.model = self.train()
        logger.info("[x] Model training ended")
        return self.model