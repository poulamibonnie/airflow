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
    def GetEvaluateModel(self, model, config: TextModelConfig, test_dataloader) -> AbstractModelsEvaluate:
        if config.model_type == "BERT":
            obj = BERTEvaluate(model, config, test_dataloader)
        elif config.model_type == "LSTM": 
            obj = LSTMEvaluate(model, config, test_dataloader)  
        else:
            raise NotImplementedError("Error the required model is not supported by The Text Operator")
        return obj
        
class BERTEvaluate(AbstractModelsEvaluate):
    def __init__(self, model, config, test_dataloader) -> None:
        super(BERTEvaluate, self).__init__()
        self.model = model
        self.data_loader = test_dataloader
        self.device = config.device
        
        ## Need to revisit
        if 'input_ids' not in test_dataloader and 'attention_mask' not in test_dataloader and 'label' not in test_dataloader: 
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
    def __init__(self, model, config, test_dataloader) -> None:
        super(LSTMEvaluate, self).__init__()
        self.model = model
        self.data_loader = test_dataloader
        self.device = config.device
        
        ## Need to revisit
        if 'input_ids' not in test_dataloader and 'attention_mask' not in test_dataloader and 'label' not in test_dataloader: 
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
        
    def evaluate(self, model, test_dataloader) -> AbstractModelsEvaluate:
        try:
            mef = ModelEvaluateFactory()
            obj = mef.GetEvaluateModel(model, self.config, test_dataloader)
            return obj.evaluate()
        except Exception as err:
            logger.error("Failed Calling the GetEvaluateModel")
            err.with_traceback()
        
        
    def run(self, model, test_dataloader):
        try:
            logger.info("Started Evaluating Model")
            self.evaluate(model, test_dataloader)
            logger.info("Model Evaluation Completed successfully")
        except Exception as err:
            logger.error("Model Evaluation Completion Failed")
            err.with_traceback()
