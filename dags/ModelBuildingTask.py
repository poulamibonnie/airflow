import torch.nn as nn 
import torch.nn.functional as func
from transformers import BertModel, DistilBertModel, DistilBertForSequenceClassification
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
            raise NotImplementedError()
        else:
            raise NotImplementedError("Error the required model is not supported by The Text Operator")

    
class BertClassifer(nn.Module, AbstractModelsBuilder):
    input_ids, attention_mask = None, None 

    def __init__(self, config: TextModelConfig) -> None:
        super(BertClassifer, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained(config.bert_model_name)
        self.drop = nn.Dropout(p=config.dropout)
        self.fc = nn.Linear(self.distilbert.config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask):
        try:
            outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0]  
            output = self.drop(pooled_output)
            return self.fc(output)
        except Exception as err:
            logger.error("Error in setting up the forward layer during build")
            err.with_traceback()
            raise
    

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