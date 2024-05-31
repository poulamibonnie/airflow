from abc import ABC, abstractmethod

# operator config custom 
from OperatorConfig import TextModelConfig

class AbstractModelsDeploy(ABC):
    @abstractmethod
    def deploy(self):
        raise NotImplementedError()

class ModelDeployFactory(object):
    def __init__(self, config: TextModelConfig) -> None: 
        self.config = config 
    
    # pass the other kwargs that is too specific for a specific model
    @staticmethod
    def GetDeployModel(self) -> AbstractModelsDeploy:
        if self.config.deploy_type == "LOCAL":
            return LocalDeploy(self.config.local_deploy_loc)
        elif self.config.deploy_type == "FTP": 
            pass
        else:
            raise NotImplementedError("Error the required model deployment is not supported")
        
class LocalDeploy(AbstractModelsDeploy):
    def __init__(self, location, **kwargs) -> None:
        super(self).__init__()
        self.location = location
    
    def deploy(self):
        pass

# implement base interface for following Builder pattern or abc class 
class ModelDeploy(object):
    def __init__(self, config: TextModelConfig) -> None:
        super().__init__()
        self.config = config

    def deploy(self) -> AbstractModelsDeploy:
        mef = ModelDeployFactory()
        obj = mef.GetDeployModel(self.config.deploy_type)
        return obj.deploy()
        
    def run(self):
        raise NotImplementedError()
