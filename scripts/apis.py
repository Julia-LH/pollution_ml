#import requests
import pandas as pd
from datetime import datetime
from joblib import load

from funcions import pipeline_processat_test

date_parser = lambda x:datetime.strptime(x, '%Y-%m-%d')
test = pd.read_csv('datasets/test_df.csv', parse_dates=['data'], date_parser=date_parser)
var_list = ['temperatura', 'humitat relativa', 'irradiància solar global', 'NO_dia']
enc = load('datasets/encoder.joblib')
scaler = load('datasets/scaler.joblib')
estimator = load('datasets/model_polucio.joblib')
print(estimator)
x_test, y_test = pipeline_processat_test(data=test, var_list=var_list, encoder=enc, scaler=scaler)
y_pred = estimator.predict(x_test)
print(y_test, y_pred)