from abc import ABC, abstractmethod
import pickle

# operator config custom 
from OperatorConfig import TextModelConfig

from AirflowDeepLearningOperatorLogger import OperatorLogger
logger = OperatorLogger.getLogger()

class AbstractModelsDeploy(ABC):
    @abstractmethod
    def deploy(self):
        raise NotImplementedError()

class ModelDeployFactory(object):
    def __init__(self, config: TextModelConfig) -> None: 
        self.config = config 
    
    def getDeployModel(self) -> AbstractModelsDeploy:
        if self.config.deployment_type == "local":
            return LocalDeploy(self.config.deployment_conf["location"])
        else:
            raise NotImplementedError("Error the required model deployment is not supported")
        
class LocalDeploy(AbstractModelsDeploy):
    def __init__(self, location, **kwargs) -> None:
        self.location = location
    
    def deploy(self, train_model):
        with open(self.location, 'wb') as f:
            pickle.dump(train_model, f)

# implement base interface for following Builder pattern or abc class 
class ModelDeploy(object):
    def __init__(self, config: TextModelConfig) -> None:
        super().__init__()
        self.config = config

    def deploy(self, train_model):
        mef = ModelDeployFactory(self.config)
        obj = mef.getDeployModel()
        obj.deploy(train_model)
        
    def run(self, train_model):
        logger.info("Started Model Deployment")
        self.deploy(train_model)
        logger.info("Model Deployment Complete")
