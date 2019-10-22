from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from numerai_pipeline.model import train, predict

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

t1 = PythonOperator(
    task_id='train',
    python_callable=train.main,
    dag=dag)

t2 = PythonOperator(
    task_id='predict',
    python_callable=predict.main,
    dag=dag)

t1 >> t2
