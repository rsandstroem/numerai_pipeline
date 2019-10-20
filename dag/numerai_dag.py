from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2015, 6, 1),
    'email': ['airflow@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
}

dag = DAG('numerai_pipeline', default_args=default_args, schedule_interval=timedelta(days=1))

t1 = BashOperator(
    task_id='train',
    bash_command='python /home/rikard/WORK/numerai_kazutsugi/src/numerai_pipeline/model/train.py',
    dag=dag)

t2 = BashOperator(
    task_id='predict',
    bash_command='python /home/rikard/WORK/numerai_kazutsugi/src/numerai_pipeline/model/predict.py',
    dag=dag)

t1 >> t2
