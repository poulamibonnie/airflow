import torch
from sklearn.metrics import accuracy_score, classification_report
from abc import ABC, abstractmethod
# logger 
from AirflowDeepLearningOperatorLogger import OperatorLogger

# operator config custom 
from OperatorConfig import TextModelConfig

logger = OperatorLogger.getLogger()

class AbstractModelsEvaluate(ABC):
    @abstractmethod
    def evaluate(self):
        raise NotImplementedError()

class ModelEvaluateFactory(object):
    def __init__(self) -> None: pass 
    
    # pass the other kwargs that is too specific for a specific model
    @staticmethod
    def GetEvaluateModel(self, model, config: TextModelConfig, **kwargs) -> AbstractModelsEvaluate:
        if config.model_type == "BERT":
            obj = BERTEvaluate(model, **kwargs)
        elif config.model_type == "LSTM": 
            obj = LSTMEvaluate(model, **kwargs)  
        else:
            raise NotImplementedError("Error the required model is not supported by The Text Operator")
        return obj
        
class BERTEvaluate(AbstractModelsEvaluate):
    def __init__(self, model, **kwargs) -> None:
        super(BERTEvaluate, self).__init__()
        self.model = model
        self.data_loader = kwargs
        self.device = 'cpu' if not torch.cuda.is_available() else 'gpu'
        
        if 'input_ids' not in kwargs and 'attention_mask' not in kwargs and 'label' not in kwargs: 
            raise RuntimeError("These params are required to evaluate model accuracy")
    
    def evaluate(self):
        self.model.eval()
        pred_level = []
        actual_label = []
        with torch.no_grad():
            for batch in self.data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                pred_level.extend(preds.cpu().tolist())
                actual_label.extend(labels.cpu().tolist())
        return accuracy_score(actual_label, pred_level), classification_report(actual_label, pred_level)  

## LSTM logic is incorrect and kept as place holder : need to revisit
class LSTMEvaluate(AbstractModelsEvaluate):
    def __init__(self, model, **kwargs) -> None:
        super(LSTMEvaluate, self).__init__()
        self.model = model
        self.data_loader = kwargs
        self.device = 'cpu' if not torch.cuda.is_available() else 'gpu'
        
        if 'input_ids' not in kwargs and 'attention_mask' not in kwargs and 'label' not in kwargs: 
            raise RuntimeError("These params are required to evaluate model accuracy")
    
    def evaluate(self):
        self.model.eval()
        pred_level = []
        actual_label = []
        with torch.no_grad():
            for batch in self.data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, preds = torch.max(outputs, dim=1)
                pred_level.extend(preds.cpu().tolist())
                actual_label.extend(labels.cpu().tolist())
        return accuracy_score(actual_label, pred_level), classification_report(actual_label, pred_level)  
    
# implement base interface for following Builder pattern or abc class 
class ModelEvaluate(object):
    def __init__(self, config) -> None:
        super().__init__()
        
    def evaluate(self, model, config: TextModelConfig, extra_config: dict) -> AbstractModelsEvaluate:
        mef = ModelEvaluateFactory()
        obj = mef.GetEvaluateModel(model=model, config=TextModelConfig, **extra_config)
        return obj.evaluate()
        
    def run(self):
        raise NotImplementedError("error the contract is not meant for this class")
