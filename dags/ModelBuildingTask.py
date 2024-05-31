import torch.nn as nn 
import torch.nn.functional as func
from transformers import BertModel
from abc import ABC, abstractmethod
from AirflowDeepLearningOperatorLogger import OperatorLogger

# operator config custom 
from OperatorConfig import TextModelConfig

class AbstractModelsBuilder(ABC):
    # forward feed nueral network layer for a new leraning layer over the base transformer or new torch def
    logger = OperatorLogger.getLogger()
    @abstractmethod
    def forward(self):
        raise NotImplementedError()

class ModelBuildFactory(object):

    def __init__(self) -> None: pass 

    # pass the other kwargs that is too specific for a specific model
    @staticmethod
    def trainModel(self, config: TextModelConfig, **kwargs) -> AbstractModelsBuilder:
        if config.model_type == "BERT":
            return BertClassifer(config, kwargs)
        elif config.model_type == "LSTM": 
            return LSTMRNN(config, kwargs)
        else:
            raise NotImplementedError("Error the required model is not supported by The Text Operator")

class LSTMRNN(nn.Module, AbstractModelsBuilder):
    extra_args = {}

    def __init__(self, config: TextModelConfig, **kwargs) -> None:
        super(LSTMRNN, self).__init__()
        self.extra_args = kwargs # pass and store some extra args specific to this model
        self.embedding = nn.Embedding(config.input_dim, config.embedding_dim)

        self.rnn = nn.LSTM(config.embedding_dim, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)

    
    def forward(self) -> None:
        if 'text' not in self.extra_args: 
            raise RuntimeError("The model require this mandatory params to work")
        embedded = self.embedding(self.extra_args['text'])
        
        output, (hidden, cell) = self.rnn(embedded)
        
        output = self.fc(hidden)
        return output

    
class BertClassifer(nn.Module, AbstractModelsBuilder):
    input_ids, attention_mask = None, None 
    extra_args = {} 

    def __init__(self, config: TextModelConfig, **kwargs) -> None:
        super(BertClassifer, self).__init__()
        if 'input_ids' not in kwargs and 'attention_mask' not in kwargs: 
            raise RuntimeError("The model require this mandatory params to work")

        self.extra_args = kwargs
        self.input_ids = kwargs['input_ids']
        self.attention_mask = kwargs['attention_mask']
        self.bert = BertModel.from_pretrained(config.pretrained_model_name)
        self.drop = nn.Dropout(p=config.dropout)
        self.output = nn.Linear(self.bert.config.hidden_size, config.num_classes)

    def forward(self):
        _, pooled_output = self.bert(
            self.input_ids,
            self.attention_mask
        )
        output = self.drop(pooled_output)
        return self.output(output)


# implement base interface for following Builder pattern or abc class 
class ModelBuilding(object):
    config: TextModelConfig
    def __init__(self, config, extra_config: dict) -> None:
        super().__init__()
        self.config = config
        self.extra_config = extra_config
    
    def build(self) -> AbstractModelsBuilder:
        factory = ModelBuildFactory.trainModel(config=self.config.model_type, **self.extra_config)
        return factory

    def run(self): raise NotImplementedError("error the contract is not meant for this class")