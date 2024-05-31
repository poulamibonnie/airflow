from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from AirflowDeepLearningOperator import AirflowDeepLearningOperator
from OperatorConfig import TextModelConfig

from airflow import DAG
from datetime import timedelta
from airflow.utils.dates import days_ago
from typing import Dict , Any


default_args = {
    'owner': 'synarcs',
    'email_on_failure': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=2),
    'start_date': days_ago(1)
}

# this is also possible to fecthced from dag config for kwargs passed as task instance
# but i need this to be generic and follow an arch design patterns 
# can be done via dataclass or pydantic class
class DagConifg:
    rootDag: DAG 
    dag_args = Dict[str, Any]


def push_args(**kwargs):
    ti = kwargs['ti']
    print(kwargs)
    inputOperatorConfig: TextModelConfig = { }
    inputOperatorConfig.model_type = 'BERT'
    '''
        TODO: 
            will populate all fields for input in python code raw 
            read it from config file (Later once first integration run is done )
    '''

    ti.xcom_push(key='model_input', value=inputOperatorConfig)

with DAG(
    dag_id="Airflow_Deep_Learning_Operator",
    description="test docker run",
    default_args=default_args,
    schedule_interval=None 
) as dag:
    base = BashOperator(task_id="init_run",
                        bash_command="echo 'Starting training the model'")
    config: DagConifg = DagConifg()
    config.rootDag = dag; config.dag_args = default_args 

    operatorInput = PythonOperator(task_id="pass_commands", python_callable=push_args)
    tensorflow_v = AirflowDeepLearningOperator(task_id="version", config=config)
    base >> operatorInput >> tensorflow_v



