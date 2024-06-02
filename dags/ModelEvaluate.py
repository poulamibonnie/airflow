import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score,recall_score, classification_report
from abc import ABC
from AirflowDeepLearningOperatorLogger import OperatorLogger
from OperatorConfig import TextModelConfig
import sys 

logger = OperatorLogger.getLogger()
class AbstractModelsEvaluate(ABC):
    @classmethod
    def evaluate(self):
        raise NotImplementedError()

class ModelEvaluateFactory(object):
    def __init__(self) -> None: pass 
    
    # pass the other kwargs that is too specific for a specific model
    @staticmethod
    def GetEvaluateModel(model: object, config: TextModelConfig, test_dataloader: object) -> AbstractModelsEvaluate:
        if config.model_type == "BERT":
            obj = BERTEvaluate(model, config, test_dataloader)
        elif config.model_type == "LSTM": 
            raise NotImplementedError()
        else:
            raise NotImplementedError("Error the required model is not supported by The Text Operator")
        return obj
        
class BERTEvaluate(AbstractModelsEvaluate):
    def __init__(self, model, config, test_dataloader) -> None:
        super(BERTEvaluate, self).__init__()
        self.model = model
        self.data_loader = test_dataloader
        self.device = config.device
        
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
            accuracy = accuracy_score(actual_label, pred_level)
            f1 = f1_score(actual_label, pred_level)
            precision = precision_score(actual_label, pred_level)
            recall = recall_score(actual_label, pred_level)
            return accuracy, f1, precision, recall
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
            logger.info("Model Evaluation Completed successfully")
            return obj.evaluate()
        except Exception as err:
            logger.error("Failed Calling the GetEvaluateModel")
            print(err) 
        
        
    def run(self, model, test_dataloader):
        try:
            logger.info("Started Evaluating Model")
            return self.evaluate(model, test_dataloader)
        except Exception as err:
            logger.error("Model Evaluation Completion Failed")