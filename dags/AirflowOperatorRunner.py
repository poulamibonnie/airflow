
from abc import ABC, abstractclassmethod
from ModelBuildingTask import Embedding
import torch 

class AbstractAirflowRunner(ABC):

    @classmethod
    def runner(config: str):
        raise NotImplementedError()


class AirflowOperatorRunner(AbstractAirflowRunner):
    config = {}

    def __init__(self, config=None) -> None:
        super().__init__()
        if config != None:  
            print('input passed to the runner is ', config)
            self.config = config
    
    def runner(self, config):
        testEmbed = Embedding() 
        print(testEmbed, torch.cuda.is_available())
        device = 'cpu' if not torch.cuda.is_available() else 'gpu'
        print('starting the model training phase for the operator', config.rootDag.dag_id)



    

