
from abc import ABC
from OperatorConfig import TextModelConfig, OperatorConfig
import torch 

class AbstractAirflowRunner(ABC):

    @classmethod
    def runner(config: str):
        raise NotImplementedError()


class AirflowOperatorRunner(AbstractAirflowRunner):
    config = {}

    def __init__(self, config:TextModelConfig =None, operatorConfig: OperatorConfig=None) \
                -> None:
        super().__init__()
        if config != None:  
            print('input passed to the runner is ', config)
            self.config = config
    
    def runner(self, config):
        device = 'cpu' if not torch.cuda.is_available() else 'gpu'
        print('starting the model training phase for the operator', config.rootDag.dag_id)
        # we need to assemble everything here to make it loose coupled 
        # and pass the values to other as required.
        
        '''
            TODO:
                We will add builder pattern by passing config:TextModelConfig  to each stage of pipeline operation 

                DataLoader (Pytorch) = DataIngestion(config)
     
                
                Model      (Pytorch)    = ModelBuilding(config, extra_config) 
                Train Model      = ModelTrainingOperator(config, Model, DataLoader)
                Torch Model Metrics = ModelEvaluate(config, Train Model , DataLoader)
                                    = ModelPredict(cofig .trained Model)
                
                SaveModel           = SaveModel(config , Train Model )

        '''

    

