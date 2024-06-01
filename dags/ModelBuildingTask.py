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
    def buildModel(config: TextModelConfig) -> AbstractModelsBuilder:
        if config.model_type == "BERT":
            model = BertClassifer(config=config).to(device=config.device)
            print(model)
            return model
        elif config.model_type == "LSTM": 
            model = LSTMRNN(config=config).to(device=config.device)
            print(model)
            return model 
        else:
            raise NotImplementedError("Error the required model is not supported by The Text Operator")

class LSTMRNN(nn.Module, AbstractModelsBuilder):

    def __init__(self, config: TextModelConfig) -> None:
        super(LSTMRNN, self).__init__()
        self.embedding = nn.Embedding(config.input_dim, config.embedding_dim)

        self.rnn = nn.LSTM(config.embedding_dim, config.hidden_dim)
        self.fc = nn.Linear(config.hidden_dim, config.output_dim)

    
    def forward(self, x, hidden) -> None:
        batch_size = x.size(0)

        embedded = self.embedding(x) 
        output, (hidden, cell) = self.rnn(embedded)
        
        output = self.fc(hidden)
        return output

    
class BertClassifer(nn.Module, AbstractModelsBuilder):
    input_ids, attention_mask = None, None 

    def __init__(self, config: TextModelConfig) -> None:
        super(BertClassifer, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_model_name)
        self.drop = nn.Dropout(p=config.dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask):
        try:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask) 
            pooled_output = outputs.pooler_output
            output = self.drop(pooled_output)
            return self.fc(output)
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
        model = ModelBuildFactory.buildModel(config=self.config)
        return model 
    
    # the extra config is a dict with model specific extra config 
    '''
        for example bert require input id's and attention mask for its forward layer 
        for example lstm requires the hidden layer for it as a input 
    '''
    def run(self): 
        logger.info("Started Model Building Task")
        model = self.build()
        logger.info("Model Building Completed successfully")
        return model 