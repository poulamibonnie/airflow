import torch
from sklearn.metrics import accuracy_score, classification_report
from abc import ABC
from AirflowDeepLearningOperatorLogger import OperatorLogger
from OperatorConfig import TextModelConfig

logger = OperatorLogger.getLogger()
class AbstractModelsEvaluate(ABC):
    @classmethod
    def evaluate(self):
        raise NotImplementedError()

class ModelEvaluateFactory(object):
    def __init__(self) -> None: pass 
    
    # pass the other kwargs that is too specific for a specific model
    @staticmethod
    def GetEvaluateModel(self, model, config: TextModelConfig) -> AbstractModelsEvaluate:
        if config.model_type == "BERT":
            obj = BERTEvaluate(model)
        elif config.model_type == "LSTM": 
            obj = LSTMEvaluate(model)  
        else:
            raise NotImplementedError("Error the required model is not supported by The Text Operator")
        return obj
        
class BERTEvaluate(AbstractModelsEvaluate):
    def __init__(self, model) -> None:
        super(BERTEvaluate, self).__init__()
        self.model = model
        self.data_loader = config.extra_config ## Need to revisit this in incorrect 
        self.device = config.device
        
        
        ## Need to revisit
        if 'input_ids' not in config.extra_config and 'attention_mask' not in config.extra_config and 'label' not in config.extra_config: 
            raise RuntimeError("These params are required to evaluate model accuracy")
    
    def evaluate(self):
        try:
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
        except Exception as err:
            logger.error("Error in evaluating model accuracy")
            err.with_traceback()

## LSTM logic is incorrect and kept as place holder : need to revisit
class LSTMEvaluate(AbstractModelsEvaluate):
    def __init__(self, model) -> None:
        super(LSTMEvaluate, self).__init__()
        self.model = model
        self.data_loader = config.extra_config ## Need to revisit
        self.device = config.device
        
        if 'input_ids' not in config.extra_config and 'attention_mask' not in config.extra_config and 'label' not in config.extra_config: 
            raise RuntimeError("These params are required to evaluate model accuracy")
    
    def evaluate(self):
        try:
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
        except Exception as err:
            logger.error("Error in evaluating model accuracy")
            err.with_traceback()
    
# implement base interface for following Builder pattern or abc class 
class ModelEvaluate(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
    def evaluate(self, model) -> AbstractModelsEvaluate:
        try:
            mef = ModelEvaluateFactory()
            obj = mef.GetEvaluateModel(model, self.config)
            return obj.evaluate()
        except Exception as err:
            logger.error("Failed Calling the GetEvaluateModel")
            err.with_traceback()
        
        
    def run(self, model):
        try:
            logger.info("Started Evaluating Model")
            self.evaluate(model)
            logger.info("Model Evaluation Completed successfully")
        except Exception as err:
            logger.error("Model Evaluation Completion Failed")
            err.with_traceback()
