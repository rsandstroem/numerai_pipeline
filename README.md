# Numerai Pipeline
This package contains an Airflow configuration for automatically submitting new results to the numer.ai competition each week.

The pipeline downloads the data from the platform, trains a number of models on the training set, prepares the submission results, and sends those to the competition server.

## Model
'Model' contains train.py and predict.py. Every individual model should be implemented in its own method in train.py, and should accept the input data as argument and return a trained model. 

Predict will create the predicted target values given a pre-trained model.

The example models in the repository are the XGB model that Numerai provides with the example files, and a very basic linear regression model. Please add your own model.

To run these tasks manually:

`train.py -m your_model`

`predict.py -m your_model`

## Transport
'Transport' contains obtain.py and submit.py, and are used to communicating with the competition server. To do that it relies on the `numerox` package.

To submit the results, you will need to provide authentication details. To simplify submission of more than one model, I used a json file with this structure:

```
{
    "user_a": {
        "id": "id provided by the website",
        "scope": "for info only",
        "key": "your api key"

    },
    "user_b": {

        "id": "id provided by the website",
        "scope": "for info only",
        "key": "your other api key"
    }
}
```

To run manually:
`submit.py -m your_model -u username`

## How to run airflow 
Airflow is used to define the pipeline, and schedule the tasks, so you can forget about it and spend time on other things.

After installing Airflow (included in requirements of this package),initialize the database: `airflow initdb`

Modify `dags_folder` in your `airflow.cfg` to point to the `dag` folder of this package. It should then pick up any DAGs that are found inside that folder.

Start the scheduler: `airflow scheduler`

Backfill some missing dates: `airflow backfill numerai_pipeline -s 2019-10-17 -e 2019-10-20`

Host the graphical interface: `airflow webserver -p 8080`