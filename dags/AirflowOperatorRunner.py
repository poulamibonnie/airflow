
from abc import ABC, abstractmethod
from OperatorConfig import TextModelConfig, OperatorConfig
from AirflowDeepLearningOperatorLogger import OperatorLogger
import torch , sys 


# all pipeline task
from DataIngestionTask import DataIngestion
from ModelBuildingTask import ModelBuilding
from ModelTraining import ModelTraining
from ModelEvaluate import ModelEvaluate
# from ModelPredict import ModelPredict 
from ModelDeploy import ModelDeploy

logger = OperatorLogger.getLogger()

class AbstractAirflowModelRunner(ABC):
    
    @abstractmethod
    def runner(self):
        raise NotImplementedError("Abstract Implementation") 

class AbstractAirflowModelBuilder(ABC):

    @abstractmethod
    def ingest_data(self) -> None: raise NotImplementedError()

    @abstractmethod 
    def build_model(self) -> None: raise NotImplementedError()

    @abstractmethod
    def train_model(self) -> None: raise NotImplementedError()

    @abstractmethod
    def evaluate_mode(self) -> None: raise NotImplementedError()

    @abstractmethod
    def predict_model(self) -> None: raise NotImplementedError()

    @abstractmethod
    def save_model(self) -> None: raise NotImplementedError()

class NLPTextDNN(object):
    data: object 
    buildModel: object
    trainModel: object
    model_metrics: object
    predict_object: object
    savedModelPath: object

'''
    The root Airflow Operator builder
    This can build entire end to end data pipeline from our custom Airflow operator for any deep learning algorithm 
'''

class AirflowTextDnnModelBuilder(AbstractAirflowModelBuilder):

    def __init__(self, config: TextModelConfig) -> None:
        super(AirflowTextDnnModelBuilder, self).__init__()
        self.textDNN = NLPTextDNN() # i have kept this simple it should be a interface to extend create class 
        # however any ml or deep learnign mdel has all of these 4 metrics 
        self.config = config 
        self.dataIngestObject = DataIngestion(config=self.config)
        self.modelBuildObject = ModelBuilding(config=self.config)
        self.modelTrainObject = ModelTraining(config=self.config)
        self.modelEvaluateObject = ModelEvaluate(config=self.config)
        # self.modelPredictObject = ModelPredict(config=self.config) 
        self.modelDeployObject = ModelDeploy(config=self.config)

    # can be object
    ''' we need to make sure to keep in consistent to always call run and then it internally calls the required metho
        to loosely couple proc calls 
    '''
    def ingest_data(self) -> object:
        self.textDNN.data = self.dataIngestObject.run()
        return self 
    
    def build_model(self) -> object:
        self.textDNN.buildModel = self.modelBuildObject.run()
        return self

    def train_model(self) -> object:
        self.textDNN.trainModel = self.modelTrainObject.run()
        return self
    
    def evaluate_mode(self) -> object:
        self.textDNN.model_metrics = self.modelEvaluateObject.run()
        return self 
    
    def predict_model(self) -> object:
        self.textDNN.predict_object = self.modelPredictObject.run()
        return self 

    def save_model(self) -> object:
        self.textDNN.model_metrics = self.modelEvaluateObject.run()
        return self 
        

class AirflowOperatorRunner(AbstractAirflowModelRunner):

    def __init__(self, config:TextModelConfig =None, operatorConfig: OperatorConfig=None) \
                -> None:
        super().__init__()
        if config != None:  
            print('input passed to the runner is ', config)
            self.config = config
            self.operatorConfig = operatorConfig
        else:
            print('no config value found Operator Exit')
            sys.exit(1) 

    def runner(self):
        device = 'cpu' if not torch.cuda.is_available() else 'gpu'
        # we need to assemble everything here to make it loose coupled 
        # and pass the values to other as required.
        print(self.config)
        '''
            TODO:
                We will add builder pattern by passing config:TextModelConfig  to each stage of pipeline operation 

                DataLoader (Pytorch) = DataIngestion(config)
     
                
                Model      (Pytorch)    = ModelBuilding(config) 
                Train Model      = ModelTrainingOperator(config, Model, DataLoader)
                Torch Model Metrics = ModelEvaluate(config, Train Model , DataLoader)
                                    = ModelPredict(cofig .trained Model)
                
                SaveModel           = SaveModel(config , Train Model )

        '''

    

