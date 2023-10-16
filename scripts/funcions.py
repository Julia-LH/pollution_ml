import pandas as pd
import haversine as hs
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, Normalizer
from sklearn.model_selection import GridSearchCV
from joblib import dump


def calcula_distancia(punt_polucio, punts_meteo):
    distancies = {'codi':[], 'nom':[], 'georef_meteo':[], 'distancia':[]}
    for index, row in punts_meteo.iterrows():
        distancies['codi'].append(row['codi_estacio'])
        distancies['nom'].append(row['nom_estacio'])
        distancies['georef_meteo'].append(row['georef_meteo'])
        distancies['distancia'].append(hs.haversine(punt_polucio, row['georef_meteo']))
    distancies_df = pd.DataFrame(distancies)
    return distancies_df[distancies_df['distancia']==distancies_df['distancia'].min()].loc[0:].values.flatten().tolist()

def omplir_buits(df, var):
    df = df.sort_values(by=['data'], axis=0)
    df[var] = df[var].fillna((df[var].shift() + df[var].shift(-1))/2)
    df[var] = df[var].fillna(method='ffill')
    df[var] = df[var].fillna(method='bfill')
    return df

def split_train_test(estacio_df, train_percent):
    min_date = estacio_df.data.min()
    max_date = estacio_df.data.max()
    time_between = max_date - min_date
    train_cutoff = min_date + train_percent*time_between
    train = estacio_df[estacio_df.data <= train_cutoff]
    test = estacio_df[estacio_df.data > train_cutoff]
    return train, test

def loop_fill_split(df, var_list):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for estacio in set(df['nom_estacio']):
        estacio_df = df.loc[df['nom_estacio'] == estacio, :]
        for var in var_list:
            estacio_df = omplir_buits(df=estacio_df, var=var)

        train, test = split_train_test(estacio_df, 0.75)
        train_df = pd.concat([train_df, train], ignore_index=True)
        test_df = pd.concat([test_df, test], ignore_index=True)
    
    return train_df, test_df

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
        estacio_df = data.loc[data['nom_estacio'] == estacio, :]
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
    # dump(enc, 'datasets/encoder.joblib')
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
    # dump(scaler, 'datasets/scaler.joblib')
    return  num, scaler

def transform_num(data, scaler):
    num = data.select_dtypes(include=np.number)
    scaled = scaler.transform(num)
    return pd.DataFrame(scaled) 

def grid_search_cv(xtrain, ytrain, models_params):
    models_params = models_params
    results = {'score' : [], 'best_estimator' : []}
    for model in models_params.keys():
        grid = GridSearchCV(estimator=models_params[model]['name'], param_grid=models_params[model]['params'], scoring='r2')
        grid.fit(X=xtrain, y=ytrain)
        results['score'].append(grid.best_score_)
        results['best_estimator'].append(grid.best_estimator_)
        best_score, best_estimator = [(score, estimator) for score, estimator in zip(results['score'], results['best_estimator']) if score==min(results['score'])][0]
    return best_score, best_estimator


def pipeline_processat_train(data, var_list, model_params):
    data = loop_dies_previs(data=data, var_list=var_list)
    x_train, y_train = split_target(data=data)
    cat, enc = fit_transform_dummies(data=x_train)
    num, scaler = fit_transform_num(data=x_train)
    X = pd.concat([data.data, cat, num], axis=1, ignore_index=True)
    x_train = X.set_index([0], drop=True)
    best_score, best_estimator = grid_search_cv(xtrain=x_train, ytrain=y_train, models_params=model_params)
    # dump(best_estimator, 'datasets/best_estimator.joblib')
    return x_train, y_train, enc, scaler, best_score, best_estimator

def pipeline_processat_test(data, var_list, encoder, scaler, best_estimator, r2_score):
    data = loop_dies_previs(data=data, var_list=var_list)
    x_test, y_test = split_target(data=data)
    cat = transform_dummies(data=x_test, encoder=encoder)
    num = transform_num(data=x_test, scaler=scaler)
    x_test = pd.concat([data.data, cat, num], axis=1, ignore_index=True)
    x_test.set_index([0], drop=True, inplace=True)
    y_pred = best_estimator.predict(x_test)
    r2_score = r2_score(y_pred, y_test)
    return r2_score

def input_data(est_atrib, date, meteo):
    
    df = pd.DataFrame(columns=['nom_estacio', 'tipus_estacio', 'area_urbana', 'altitud', 'latitud', 'longitud'])
    



def prediction(data, encoder, scaler, best_estimator):
    cat = transform_dummies(data=data, encoder=encoder)
    num = transform_num(data=data, scaler=scaler)
    X = pd.concat([data.data, cat, num], axis=1, ignore_index=True)
    X.set_index([0], drop=True, inplace=True)
    return best_estimator.predict(X)