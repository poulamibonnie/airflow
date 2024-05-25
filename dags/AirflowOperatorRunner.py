
from abc import ABC, abstractclassmethod

class AbstractAirflowRunner(ABC):

    @classmethod
    def runner(config: str):
        raise NotImplementedError()


class AirflowOperatorRunner(AbstractAirflowRunner):
    
    def __init__(self) -> None:
        super().__init__()

    
    def runner(self, config):
        print('starting the model training phase for the operator', config.rootDag.dag_id)



    

