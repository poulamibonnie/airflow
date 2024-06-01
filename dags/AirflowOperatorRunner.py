
from abc import ABC, abstractmethod
from OperatorConfig import TextModelConfig, OperatorConfig
from AirflowDeepLearningOperatorLogger import OperatorLogger
from dataclasses import make_dataclass, dataclass
import torch , sys 


# all pipeline task
from DataIngestionTask import DataIngestion
from ModelBuildingTask import ModelBuilding
from ModelTraining import ModelTraining
from ModelEvaluate import ModelEvaluate
# from ModelPredict import ModelPredict 
# from ModelDeploy import ModelDeploy

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

@dataclass
class DataIngestionOutput(object):
    tokenizer: object = None 
    train_dataset: object = None 
    test_dataset: object = None 
    train_dataloader: object = None
    test_dataloader: object = None

@dataclass
class NLPTextDNN(object):
    data: DataIngestionOutput = None 
    buildModel: object = None
    trainModel: object = None 
    model_metrics: object = None 
    predict_object: object = None 
    savedModelPath: object = None 

'''
    The root Airflow Operator builder
    This can build entire end to end data pipeline from our custom Airflow operator for any deep learning algorithm 
'''

class AirflowTextDnnModelBuilder(AbstractAirflowModelBuilder):

    def __init__(self, config: dict) -> None:
        super(AirflowTextDnnModelBuilder, self).__init__()
        self.textDNN = NLPTextDNN() # i have kept this simple it should be a interface to extend create class 
        # however any ml or deep learnign mdel has all of these 4 metrics 
        config = make_dataclass(
            "TextModelConfig", ((k, type(v)) for k, v in config.items())
        )(**config)
        print(type(config))
        self.config = config 
        self.dataIngestObject = DataIngestion(config=self.config)
        self.modelBuildObject = ModelBuilding(config=self.config)
        self.modelTrainObject = ModelTraining(config=self.config)
        # self.modelEvaluateObject = ModelEvaluate(config=self.config)
        # # self.modelPredictObject = ModelPredict(config=self.config) 
        # self.modelDeployObject = ModelDeploy(config=self.config)

    # can be object
    ''' we need to make sure to keep in consistent to always call run and then it internally calls the required metho
        to loosely couple proc calls 
    '''
    def ingest_data(self) -> object:
        tokenizer, train_dataset, test_dataset, train_dataloader, test_dataloader = \
                        self.dataIngestObject.run()
        self.textDNN.data = DataIngestionOutput()
        self.textDNN.data.tokenizer = tokenizer 
        self.textDNN.data.train_dataset = train_dataset
        self.textDNN.data.test_dataset = test_dataset
        self.textDNN.data.train_dataloader = train_dataloader
        self.textDNN.data.test_dataloader = test_dataloader
        return self
    
    def build_model(self) -> object:
        self.textDNN.buildModel = self.modelBuildObject.run()
        return self

    def train_model(self) -> object:
        self.textDNN.trainModel = self.modelTrainObject.run(model=self.textDNN.buildModel,
                                                                data_loader=self.textDNN.data.train_dataloader)
        return self 
    
    def evaluate_mode(self) -> object:
        pass 
        # self.textDNN.model_metrics = self.modelEvaluateObject.run()
        # return self 
    
    def predict_model(self) -> object:
        pass
        # self.textDNN.predict_object = self.modelPredictObject.run()
        # return self 

    def save_model(self) -> object:
        pass 
        # self.textDNN.model_metrics = self.modelEvaluateObject.run()
        # return self 
        

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
        builder = AirflowTextDnnModelBuilder(self.config)
        builder.ingest_data()
        builder.build_model()
        builder.train_model()
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

    

