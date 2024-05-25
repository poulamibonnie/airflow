from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from AirflowDeepLearningOperator import AirflowDeepLearningOperator


from airflow import DAG 
from datetime import timedelta
from airflow.utils.dates import days_ago
from typing import Dict , Any


import tensorflow as tf 

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

with DAG(
    dag_id="Airflow_Deep_Learning_Operator",
    description="test docker run",
    default_args=default_args,
    schedule_interval=None 
) as dag:
    base = BashOperator(task_id="bash_run",
                        bash_command="echo 'Starting training the model'")
    config: DagConifg = DagConifg()
    config.rootDag = dag; config.dag_args = default_args 

    tensorflow_v = AirflowDeepLearningOperator(task_id="version", config=config)
    base >> tensorflow_v



