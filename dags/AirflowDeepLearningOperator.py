from typing import Any
from airflow.models.baseoperator import BaseOperator
from airflow.utils.context import Context
from AirflowOperatorRunner import AirflowOperatorRunner 
from OperatorConfig import OperatorConfig

class AirflowDeepLearningOperator(BaseOperator):
    task_instance = None

    def __init__(self, operatorConfig,**kwargs) -> None:
        super().__init__(**kwargs)
        self.operatorConfig = operatorConfig


    def pre_execute(self, context: Any):
        return super().pre_execute(context)
    

    def execute(self, context: Context) -> Any:
        print('custom operator kwargs', context)
        self.task_instance = context['task_instance']
        operatorConfig: OperatorConfig = OperatorConfig()
        operatorConfig.dag_id = self.operatorConfig['dag_id']; operatorConfig.task_id = self.operatorConfig['version']
        ops = AirflowOperatorRunner(config=self.task_instance.xcom_pull(task_ids="pass_commands", key="model_input"))
        ops.runner(config=self.config, extraConfig=operatorConfig)
