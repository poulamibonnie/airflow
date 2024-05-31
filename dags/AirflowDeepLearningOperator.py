from typing import Any
from airflow.models.baseoperator import BaseOperator
from airflow.utils.context import Context
from AirflowOperatorRunner import AirflowOperatorRunner 

class AirflowDeepLearningOperator(BaseOperator):
    task_instance = None

    def __init__(self, config: Any, **kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config


    def pre_execute(self, context: Any):
        return super().pre_execute(context)
    

    def execute(self, context: Context) -> Any:
        print('custom operator kwargs', context)
        self.task_instance = context['task_instance']
        ops = AirflowOperatorRunner(config=self.task_instance.xcom_pull(task_ids="pass_commands", key="input"))
        ops.runner(config=self.config)
