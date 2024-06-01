import torch
from sklearn.metrics import accuracy_score, classification_report
from abc import ABC
from OperatorConfig import TextModelConfig
from AirflowDeepLearningOperatorLogger import OperatorLogger

logger = OperatorLogger.getLogger()

class AbstractModelsPredict(ABC):
    @classmethod
    def predict(self):
        raise NotImplementedError()
    
class ModelPredictFactory(object):
    def __init__(self) -> None: pass 
    
    # pass the other kwargs that is too specific for a specific model
    @staticmethod
    def GetPredictionModel(self, config: TextModelConfig, model, tokenizer) -> AbstractModelsEvaluate:
        if config.model_type == "BERT":
            obj = BERTPredict(model, tokenizer, config)
        elif config.model_type == "LSTM": 
            obj = LSTMPredict(model, tokenizer, config)
        else:
            raise NotImplementedError("Error the required model is not supported by The Text Operator")
        return obj

class BERTPredict(AbstractModelsPredict):
    def __init__(self, model, tokenizer, config) -> None:
        super(BERTPredict, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.input_text = config.input_text_string
        self.max_length = 128
    
    def predict(self):
        try:
            model.eval()
            encoding = tokenizer(self.input_text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
            return "positive" if preds.item() == 1 else "negative"
        except Exception as err:
            logger.error("Failed in the Predict Method")
            err.with_traceback()
        
## This is incorrect and needs to be revisited
class LSTMPredict(AbstractModelsPredict):
    def __init__(self, model, tokenizer, config) -> None:
        super(LSTMPredict, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = config.device
        self.input_text = config.input_text_string
        self.max_length = 128
    
    def predict(self):
        try:
            model.eval()
            encoding = tokenizer(self.input_text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
            return "positive" if preds.item() == 1 else "negative"
        except Exception as err:
            logger.error("Failed in the Predict Method")
            err.with_traceback()

# implement base interface for following Builder pattern or abc class 
class ModelPredict(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
    def predict(self, model, tokenizer) -> AbstractModelsEvaluate:
        try:
            mef = ModelPredictFactory()
            device = getDevice()
            obj = mef.GetPredictionModel(config: TextModelConfig, model, tokenizer)
            return obj.predict()
        except Exception as err:
            logger.error("Failed Calling the GetPredictionModel")
            err.with_traceback()
        
    def run(self):
        try:
            logger.info("Started Model Predictions")
            self.predict(model, tokenizer)
            logger.info("Model Predictions Completed successfully")
        except Exception as err:
            logger.error("Model Predictions Completion Failed")
            err.with_traceback()
    
