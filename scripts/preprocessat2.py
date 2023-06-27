import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, Normalizer
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor, cv, DMatrix

from preprocessat1 import var_list, train_df, test_df



#from config import file
file='datasets/train_df.csv'

def crear_dies_previs(data, var):
    var_previ1 = var + '_previ1'
    var_previ2 = var + '_previ2'
    data = data.sort_values(by=['data'], axis=0, ignore_index=True)
    data[var_previ1] = data[var].shift(periods=-1)
    data[var_previ1] = data[var_previ1].fillna(method='ffill')
    data[var_previ2] = data[var_previ1].shift(periods=-1)
    data[var_previ2] = data[var_previ2].fillna(method='ffill')
    return data

def loop_dies_previs(data, var_list):
    df_model = pd.DataFrame()
    for estacio in set(data['nom_estacio']):
        estacio_df = data.loc[data['nom_estacio'] == estacio]
        #estacio_df = data[data['nom_estacio']==estacio]
        for var in var_list:
            estacio_df = crear_dies_previs(data=estacio_df, var=var)            
        df_model = pd.concat([df_model, estacio_df], ignore_index=True)
    return df_model

def split_target(data):
    y = data['NO_dia']
    x = data.drop('NO_dia', axis=1)
    return x, y

def fit_transform_dummies(data):
    enc = OrdinalEncoder()
    cat = data.select_dtypes(include=object)
    cat = pd.DataFrame(enc.fit_transform(cat))
    return cat, enc

def transform_dummies(data, encoder):
    enc = encoder
    cat = data.select_dtypes(include=object)
    return pd.DataFrame(enc.transform(cat))

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

def cross_validation(xtrain, ytrain):
    param = {'max_depth':2, 'eta':1, 'objective':'reg:squarederror'}
    num_round = 2
    xgtrain = DMatrix(xtrain.values, ytrain.values)
    cv_ = cv(params=param, dtrain=xgtrain, num_boost_round=num_round, nfold=5, seed=0)
    return cv_

def grid_search_cv(xtrain, ytrain):
    param = {'max_depth':[1, 2, 3, 4, 5, 6], 'eta':[1, 2, 3, 4]}
    grid = GridSearchCV(XGBRegressor(), param)
    grid.fit(xtrain, ytrain)
    return grid.best_params_

def pipeline_processat_train(data):
    print(data.info())
    data = loop_dies_previs(data=data, var_list=var_list)
    print(data.info())
    x_train, y_train = split_target(data=data)
    cat, enc = fit_transform_dummies(data=x_train)
    num, scaler = fit_transform_num(data=x_train)
    X = pd.concat([data.data, cat, num], axis=1, ignore_index=True)
    x_train = X.set_index([0], drop=True)
    best_params = grid_search_cv(xtrain=x_train, ytrain=y_train)
    cv_ = cross_validation(xtrain=x_train, ytrain=y_train)
    return x_train, y_train, enc, scaler, cv_, best_params

def pipeline_processat_test(data, encoder, scaler):
    print(data.info())
    data = loop_dies_previs(data=data, var_list=var_list)
    print(data.info())
    x_test, y_test = split_target(data=data)
    cat = transform_dummies(data=x_test, encoder=encoder)
    num = transform_num(data=x_test, scaler=scaler)
    X = pd.concat([data.data, cat, num], axis=1, ignore_index=True)
    return X.set_index([0], drop=True), y_test


def model_training(train, test):
    x_train, y_train, encoder, scaler, cv_, best_params = pipeline_processat_train(data=train)
    forecast = XGBRegressor() #       forecast = ForecasterAutoreg(regressor = XGBRegressor())
    forecast.fit(x_train, y_train)
    x_test, y_test = pipeline_processat_test(data=test, encoder=encoder, scaler=scaler)
    y_pred = forecast.predict(x_test)
    return y_pred, y_test, cv_, best_params



if __name__ == '__main__':
    y_pred, y_test, cv_, best_params = model_training(train=train_df, test=test_df)
    print("cross grid")
    print(best_params)
    
