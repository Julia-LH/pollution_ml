import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, Normalizer
from skforecast import ForecasterAutoreg
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score

from processat_df import var_list, train_df, test_df



#from config import file
file='datasets/train_df.csv'

def crear_dies_previs(data, var):
    var_previ1 = var + '_previ1'
    var_previ2 = var + '_previ2'
    data[var_previ1] = data.sort_values(by=['data'], axis=0)[var].shift(periods=-1)
    data[var_previ1].fillna(method='ffill', inplace=True)
    data[var_previ2] = data.sort_values(by=['data'], axis=0)[var_previ1].shift(periods=-1)
    data[var_previ2].fillna(method='ffill', inplace=True)
    return data

def loop_dies_previs(data, var_list):
    df_model = pd.DataFrame()
    for estacio in set(data['nom_estacio']):
        estacio_df = data[data['nom_estacio']==estacio]
        for var in var_list:
            estacio_df = crear_dies_previs(data=estacio_df, var=var)            
        df_model = pd.concat([df_model, estacio_df], ignore_index=True)
    return df_model

def split_target(data):
    y = data['NO_dia']
    x = data.drop('NO_dia', axis=1)
    return x, y

'''
def get_dummies(data):
    enc = OrdinalEncoder()
    cat = data.select_dtypes(include=object)
    if data==train_df:
        enc.fit(cat)
    return pd.DataFrame(enc.transform(cat))'''

def fit_transform_dummies(data):
    enc = OrdinalEncoder()
    cat = data.select_dtypes(include=object)
    cat = pd.DataFrame(enc.fit_transform(cat))
    return cat, enc

def transform_dummies(data, encoder):
    enc = encoder
    cat = data.select_dtypes(include=object)
    return pd.DataFrame(enc.transform(cat))

'''
def scale_num(data):
    num = data.select_dtypes(include=np.number)
    if data==train_df:
        scaler = Normalizer().fit(num)
    scaled = scaler.transform(num)
    return pd.DataFrame(scaled) '''

def fit_transform_num(data):
    num = data.select_dtypes(include=np.number)
    scaler = Normalizer().fit(num)
    scaled = scaler.transform(num)
    num = pd.DataFrame(scaled)
    return  num, scaler

def transform_num(data, scaler):
    num = data.select_dtypes(include=np.number)
    scaled = scaler.transform(num)
    return pd.DataFrame(scaled) 


def pipeline_processat_train(data):
    data = loop_dies_previs(data=data, var_list=var_list)
    x, y = split_target(data=data)
    cat, enc = fit_transform_dummies(data=x)
    num, scaler = fit_transform_num(data=x)
    X = pd.concat([data.data, cat, num], axis=1, ignore_index=True)
    x_train = X.set_index([0], drop=True)
    return x_train, y, enc, scaler

def pipeline_processat_test(data, encoder, scaler):
    data = loop_dies_previs(data=data, var_list=var_list)
    x, y = split_target(data=data)
    cat = transform_dummies(data=x, encoder=encoder)
    num = transform_num(data=x, scaler=scaler)
    X = pd.concat([data.data, cat, num], axis=1, ignore_index=True)
    return X.set_index([0], drop=True), y

#

def model_training(train, test):
    x_train, y_train, encoder, scaler = pipeline_processat_train(data=train)
    forecast = XGBRegressor() #       forecast = ForecasterAutoreg(regressor = XGBRegressor())
    forecast.fit(x_train, y_train)
    x_test, y_test = pipeline_processat_test(data=test, encoder=encoder, scaler=scaler)
    y_pred = forecast.predict(x_test)
    #accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)#, normalize=False
    return y_pred, y_test#, accuracy



if __name__ == '__main__':
    y_pred, y_test = model_training(train=train_df, test=test_df) #, accuracy
    #print("accuracy: {}".format(accuracy))
    accuracy_score(y_true=y_test, y_pred=y_pred)
    print("y_pred:")
    print(y_pred)
    print("y_test:")
    print(y_test)
    
