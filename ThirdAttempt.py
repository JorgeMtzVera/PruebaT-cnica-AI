"""
Third try DAG
"""
from datetime import timedelta

import airflow
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os

#Import connection package to POSTGRESQL
from sqlalchemy import create_engine

#Default arguments of the DAG
default_args = {
    'owner': 'jorgemtzvera',
    'depends_on_past': False,
    'start_date': airflow.utils.dates.days_ago(1), 
    'email': ['jm.trabajos@gmail.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2)
}

#Define the functions which will inclu
def extract_data():
    """
    1. We set our working directory.
    2. We choose our relevant variables.
    3. Separate them into a training and a validation set.
    4. Save them into different .csv files.
    """
    os.chdir("/opt/airflow")
    archivo       = os.getcwd() + '/train.csv'
    bicicletas_df = pd.read_csv(archivo, index_col='datetime', parse_dates=True);
    engine        = create_engine('postgresql://postgres:contrasinal@192.168.0.1:5432/test');
    
    columnas_relevantes = ['temp', 'season', 'weather', 'humidity']
    parametros_ppales   = bicicletas_df[columnas_relevantes]
    bicicletas_usadas   = bicicletas_df["count"]
    
    
    parametros_ppales_train, parametros_ppales_test, bicicletas_usadas_train, bicicletas_usadas_test = train_test_split(parametros_ppales, bicicletas_usadas, random_state=21)
    
    
    parametros_ppales_train.to_csv("parametros_ppales_train.csv")
    parametros_ppales_test.to_csv("parametros_ppales_test.csv")
    bicicletas_usadas_train.to_csv("bicicletas_usadas_train.csv")
    bicicletas_usadas_test.to_csv("bicicletas_usadas_test.csv")
    
    parametros_ppales_train.to_sql("parametros_ppales_train", con = engine,
                                   if_exists = 'append')
    parametros_ppales_test.to_sql("parametros_ppales_test", con = engine,
                                   if_exists = 'append')
    bicicletas_usadas_train.to_sql("bicicletas_usadas_train", con = engine,
                                   if_exists = 'append')
    bicicletas_usadas_test.to_sql("bicicletas_usadas_test", con = engine,
                                   if_exists = 'append')
    
    
    print("Los archivos fueron guardados correctamente")

def train_model():
    """
    1. We load the training data.
    2. We obtain linear regression coefficients of a model.
    3. Save the model in a pickle file.
    """
    os.chdir("/opt/airflow")
    
    archivo_1               = os.getcwd() + '/parametros_ppales_train.csv'
    parametros_ppales_train = pd.read_csv(archivo_1, index_col='datetime', parse_dates=True);
    
    archivo_2               = os.getcwd() + '/bicicletas_usadas_train.csv'
    bicicletas_usadas_train = pd.read_csv(archivo_2, index_col='datetime', parse_dates=True);
    
    
    linreg = LinearRegression()
    linreg.fit(parametros_ppales_train, bicicletas_usadas_train)
    
    pickle.dump(linreg, open('modelo.p','wb'))
    
    
def validation():
    """
    1. We load the model from the pickle file.
    2. We load the tests from the .csv files.
    4. Save the predictions in .csv and in the POSTGRESQL database.
    5. We print metric values.
    """
    os.chdir("/opt/airflow")
    engine = create_engine('postgresql://postgres:contrasinal@192.168.0.1:5432/test');


    modelo_pred            = pickle.load(open('modelo.p','rb'))
    
    archivo_4              = os.getcwd() + '/parametros_ppales_test.csv'
    parametros_ppales_test = pd.read_csv(archivo_4, index_col='datetime', parse_dates=True);
    
    archivo_5              = os.getcwd() + '/bicicletas_usadas_test.csv'
    bicicletas_usadas_test = pd.read_csv(archivo_5, index_col='datetime', parse_dates=True);
    
    y_pred    = modelo_pred.predict(parametros_ppales_test)
    
    df_y_pred = pd.DataFrame(y_pred.tolist(),columns=['Count'])
    
    df_y_pred.to_csv("prediction.csv")
    df_y_pred.to_sql("prediction", con = engine, if_exists = 'append')
    
    
    
def plotting():
    """
    1. Plot some of the results.
    """    
    os.chdir("/opt/airflow")
    modelo_pred            = pickle.load(open('modelo.p','rb'))
    
    archivo_6              = os.getcwd() + '/parametros_ppales_test.csv'
    parametros_ppales_test = pd.read_csv(archivo_6, index_col='datetime', parse_dates=True);
    
    archivo_7              = os.getcwd() + '/bicicletas_usadas_test.csv'
    bicicletas_usadas_test = pd.read_csv(archivo_7, index_col='datetime', parse_dates=True);
    
    y_pred    = modelo_pred.predict(parametros_ppales_test)
    
    df_y_pred = pd.DataFrame(y_pred.tolist(),columns=['Count'])
    
    plt.scatter(parametros_ppales_test['temp'],bicicletas_usadas_test ,  color='gray',label="Data")
    plt.scatter(parametros_ppales_test['temp'], y_pred, color='red',label="Prediction")
    plt.xlabel("Temperature (C)")
    plt.ylabel("Number of Rides")
    plt.legend(loc='upper left')
    plt.savefig('PLOT.pdf')


# Create the DAG
with DAG(
    dag_id="Third_DAG",
    default_args=default_args,
    description='Tercer intento de utilizar un DAG',
    schedule_interval='@hourly',
    
    catchup = False
    ) as f:

# Escribimos las tareas
    
    TareaM1 = BashOperator(
            task_id         ='Start',
            bash_command = "pwd"
    )


    Tarea0 = PythonOperator(
            task_id         ='Extraction',
            python_callable = extract_data
    )
    
    Tarea1 = PythonOperator(
            task_id         ='Training',
            python_callable = train_model
    )

    Tarea2 = PythonOperator(
            task_id         ='Testing',
            python_callable = validation
    )
    
    Tarea3 = PythonOperator(
            task_id         ='Plotting',
            python_callable = plotting
    )
    
    
    
#Establecemos la jerarquía entre las tareas
TareaM1 >> Tarea0 >> Tarea1 >> Tarea2 >> Tarea3
