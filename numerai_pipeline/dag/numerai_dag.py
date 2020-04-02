from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from numerai_pipeline.model import train, predict
from numerai_pipeline.transport import obtain, submit

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2019, 10, 27),
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
          schedule_interval=timedelta(days=7))

obtain = PythonOperator(
    task_id='obtain',
    python_callable=obtain.main,
    dag=dag)

train_a = PythonOperator(
    task_id='train_lgb',
    python_callable=train.train,
    op_kwargs={'model_name': 'lgb'},
    dag=dag)

train_b = PythonOperator(
    task_id='train_xgb',
    python_callable=train.train,
    op_kwargs={'model_name': 'xgb'},
    dag=dag)

predict_a = PythonOperator(
    task_id='predict_lgb',
    python_callable=predict.predict,
    op_kwargs={'model_name': 'lgb'},
    dag=dag)

predict_b = PythonOperator(
    task_id='predict_xgb',
    python_callable=predict.predict,
    op_kwargs={'model_name': 'xgb'},
    dag=dag)

submit_a = PythonOperator(
    task_id='submit_lgb',
    python_callable=submit.submit,
    op_kwargs={'model_name': 'lgb', 'user': 'rsai'},
    dag=dag)

submit_b = PythonOperator(
    task_id='submit_xgb',
    python_callable=submit.submit,
    op_kwargs={'model_name': 'xgb', 'user': 'rsai2'},
    dag=dag)

# Setting dependencies
obtain >> train_a >> predict_a >> submit_a
obtain >> train_b >> predict_b >> submit_b
