import torch
from sklearn.metrics import accuracy_score, classification_report
from abc import ABC

# operator config custom 
from OperatorConfig import TextModelConfig


class AbstractModelsPredict(ABC):
    @classmethod
    def predict(self):
        raise NotImplementedError()
    
class ModelPredictFactory(object):
    def __init__(self) -> None: pass 
    
    # pass the other kwargs that is too specific for a specific model
    @staticmethod
    def GetPredictionModel(self, config: TextModelConfig, model, tokenizer, **kwargs) -> AbstractModelsEvaluate:
        if config.model_type == "BERT":
            obj = BERTPredict(config.input_text_string, model, tokenizer, **kwargs)
        elif config.model_type == "LSTM": 
            obj = LSTMPredict(config.input_text_string, model, tokenizer, **kwargs)
        else:
            raise NotImplementedError("Error the required model is not supported by The Text Operator")
        return obj
    
###Need to revisit
class LSTMPredict(AbstractModelsPredict):
    def __init__(self, input_text, model, device, **kwargs) -> None:
        super(LSTMPredict, self).__init__()
        self.model = model
        self.data_loader = kwargs
        self.device = 'cpu' if not torch.cuda.is_available() else 'gpu'
        self.max_length = 128
        self.tokenizer = tokenizer
        self.input_text = input_text
        
        if 'input_ids' not in kwargs and 'attention_mask' not in kwargs: 
            raise RuntimeError("These params are required to evaluate model accuracy")
    
    def predict_sentiment(self):
        model.eval()
        encoding = tokenizer(self.input_text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
        return "positive" if preds.item() == 1 else "negative"
    
class BERTPredict(AbstractModelsPredict):
    def __init__(self, input_text, model, device, **kwargs) -> None:
        super(BERTPredict, self).__init__()
        self.model = model
        self.data_loader = kwargs
        self.device = 'cpu' if not torch.cuda.is_available() else 'gpu'
        self.max_length = 128
        self.tokenizer = tokenizer
        self.input_text = input_text
        
        if 'input_ids' not in kwargs and 'attention_mask' not in kwargs: 
            raise RuntimeError("These params are required to evaluate model accuracy")
    
    def predict_sentiment(self):
        model.eval()
        encoding = tokenizer(self.input_text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
        return "positive" if preds.item() == 1 else "negative"

# implement base interface for following Builder pattern or abc class 
class ModelPredict(object):
    def __init__(self, config) -> None:
        super().__init__()
        
    def predict(self, model, tokenizer, config: TextModelConfig, extra_config: dict) -> AbstractModelsEvaluate:
        mef = ModelPredictFactory()
        device = getDevice()
        obj = mef.GetPredictionModel(config: TextModelConfig, model, tokenizer, extra_config)
        return obj.predict()
        
    def run(self):
        raise NotImplementedError("error the contract is not meant for this class")
