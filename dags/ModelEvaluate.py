import torch
from sklearn.metrics import accuracy_score, classification_report
from abc import ABC

# operator config custom 
from OperatorConfig import TextModelConfig

class AbstractModelsEvaluate(ABC):
    @classmethod
    def evaluate(self):
        raise NotImplementedError()

class ModelEvaluateFactory(object):
    def __init__(self) -> None: pass 
    
    # pass the other kwargs that is too specific for a specific model
    @staticmethod
    def GetEvaluateModel(self, model, device, config: TextModelConfig, **kwargs) -> AbstractModelsEvaluate:
        if config.model_type == "BERT":
            obj = BERTEvaluate(model, device, **kwargs)
        elif config.model_type == "LSTM": 
            pass
        else:
            raise NotImplementedError("Error the required model is not supported by The Text Operator")
        return obj
        
class BERTEvaluate(AbstractModelsEvaluate):
    def __init__(self, model, device, **kwargs) -> None:
        super(BERTEvaluate, self).__init__()
        self.model = model
        self.data_loader = kwargs
        self.device = device
        
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

    def evaluate(self, model, device, config: TextModelConfig, extra_config: dict) -> AbstractModelsEvaluate:
        mef = ModelEvaluateFactory()
        obj = mef.GetEvaluateModel(model, device, config, extra_config)
        return obj.evaluate()
        
    def run(self):
        raise NotImplementedError("error the contract is not meant for this class")
