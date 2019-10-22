from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from numerai_pipeline.model import train, predict
from numerai_pipeline.transport import obtain, submit

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2019, 10, 1),
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

dag = DAG('numerai_pipeline', default_args=default_args,
          schedule_interval=timedelta(days=1))

t0 = PythonOperator(
    task_id='obtain',
    python_callable=obtain.main,
    dag=dag)

t1a = PythonOperator(
    task_id='train',
    python_callable=train.main,
    dag=dag)

t2a = PythonOperator(
    task_id='predict',
    python_callable=predict.main,
    dag=dag)

t3a = PythonOperator(
    task_id='submit',
    python_callable=submit.main,
    dag=dag)

t0 >> t1a >> t2a >> t3a
