
from abc import ABC, abstractclassmethod

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
        print('starting the model training phase for the operator', config.rootDag.dag_id)



    

