from typing import Any
from airflow.models.baseoperator import BaseOperator
from airflow.utils.context import Context
from AirflowOperatorRunner import AirflowOperatorRunner 

from typing import Any


class AirflowDeepLearningOperator(BaseOperator):


    def __init__(self, config: Any, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
    

    def pre_execute(self, context: Any):
        return super().pre_execute(context)
    

    def execute(self, context: Context) -> Any:
        ops = AirflowOperatorRunner()
        ops.runner(config=self.config)
