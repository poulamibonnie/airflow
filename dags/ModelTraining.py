import torch
from torch import nn
from transformers import BertTokenizer
from airflow.models.baseoperator import BaseOperator
from ModelBuildingTask import ModelBuildFactory, TextModelConfig

class ModelTraining(BaseOperator):
    def __init__(self, task_id, model_config, data_source, hyperparameters, **kwargs):
        super().__init__(task_id=task_id, **kwargs)
        self.model_config = model_config
        self.data_source = data_source
        self.hyperparameters = hyperparameters

    def execute(self, context):
        # Load the data
        train_dataloader = self.data_source['train']

        # Build the model
        tokenizer = BertTokenizer.from_pretrained(self.model_config.pretrained_model_name)
        extra_config = {
            'input_ids': train_dataloader.dataset['input_ids'],
            'attention_mask': train_dataloader.dataset['attention_mask']
        }
        model_builder = ModelBuildFactory.trainModel(self.model_config, **extra_config)
        model = model_builder.build(self.model_config, extra_config)

        # Define the optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=self.hyperparameters['learning_rate'])
        loss_fn = nn.CrossEntropyLoss()

        # Training loop
        for epoch in range(self.hyperparameters['epochs']):
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