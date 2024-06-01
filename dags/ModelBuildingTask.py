import torch.nn as nn 
import torch.nn.functional as func
from transformers import BertModel
from abc import ABC, abstractmethod
from AirflowDeepLearningOperatorLogger import OperatorLogger

logger = OperatorLogger.getLogger()
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
    def buildModel(self, config: TextModelConfig) -> AbstractModelsBuilder:
        if config.model_type == "BERT":
            return BertClassifer(config)
        elif config.model_type == "LSTM": 
            return LSTMRNN(config)
        else:
            raise NotImplementedError("Error the required model is not supported by The Text Operator")

class LSTMRNN(nn.Module, AbstractModelsBuilder):

    def __init__(self, config: TextModelConfig) -> None:
        super(LSTMRNN, self).__init__()
        self.extra_args = config.extra_config # pass and store some extra args specific to this model
        self.embedding = nn.Embedding(config.input_dim, config.embedding_dim)

        self.rnn = nn.LSTM(config.embedding_dim, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)

    
    def forward(self) -> None:
        if 'text' not in self.extra_args or self.extra_args.text == None: 
            logger.error("[x] The model build task called without appropriate model parameters")
            raise RuntimeError("The model require this mandatory params to work")
        embedded = self.embedding(self.extra_args['text'])
        
        output, (hidden, cell) = self.rnn(embedded)
        
        output = self.fc(hidden)
        return output

    
class BertClassifer(nn.Module, AbstractModelsBuilder):
    input_ids, attention_mask = None, None 

    def __init__(self, config: TextModelConfig) -> None:
        super(BertClassifer, self).__init__()
        if 'input_ids' not in config.extra_config and 'attention_mask' not in config.extra_config: 
            logger.error("[x] The model build task called without appropriate model parameters")
            raise RuntimeError("The model require this mandatory params to work")
        
        if 'input_ids' in config.extra_config and 'attention_mask' in config.extra_config \
                and config.extra_config.input_ids == None and config.extra_config.attention_mask == None:
            logger.error("[x] The model build task called without appropriate model parameters")
            raise RuntimeError("[x] The model require this mandatory params to work")

        self.extra_args = config.extra_config
        self.input_ids = config.extra_config['input_ids']
        self.attention_mask = config.extra_config['attention_mask']
        self.bert = BertModel.from_pretrained(config.pretrained_model_name)
        self.drop = nn.Dropout(p=config.dropout)
        self.output = nn.Linear(self.bert.config.hidden_size, config.num_classes)

    def forward(self):
        try:
            _, pooled_output = self.bert(
                self.input_ids,
                self.attention_mask
            )
            output = self.drop(pooled_output)
            return self.output(output)
        except Exception as err:
            logger.error("Error in setting up the forward layer during build")
            err.with_traceback()

# implement base interface for following Builder pattern or abc class 
class ModelBuilding(object):
    config: TextModelConfig
    def __init__(self, config: TextModelConfig) -> None:
        super().__init__()
        self.config = config
    
    def build(self) -> AbstractModelsBuilder:
        factory = ModelBuildFactory.buildModel(config=self.config)
        return factory

    def run(self): 
        logger.info("Started Model Building Task")
        self.build()
        logger.info("Model Building Completed successfully")