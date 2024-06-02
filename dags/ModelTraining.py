import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from ModelBuildingTask import (
    ModelBuildFactory,
    TextModelConfig,
    BertClassifer,
    AbstractModelsBuilder
)
from transformers import get_linear_schedule_with_warmup
from AirflowDeepLearningOperatorLogger import OperatorLogger
from transformers import Trainer, TrainingArguments

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
            return NotImplementedError(config)
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
        for epoch in range(1, self.config.num_epochs + 1):
            print(f"Epoch {epoch}/{self.config.num_epochs}")
            model.train()
            total_loss = 0
            for batch in train_dataloader:
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