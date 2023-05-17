import pandas as pd
import haversine as hs


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
    #estacio_df = polucio_df[polucio_df['nom estacio']==estacio]
    df.sort_values(by=['data'], axis=0, inplace=True)
    df[var].fillna((df[var].shift() + df[var].shift(-1))/2, inplace=True)
    df[var].fillna(method='ffill', inplace=True)
    df[var].fillna(method='bfill', inplace=True)
    return df

def crear_dies_previs(df, var):
    var_previ1 = var + '_previ1'
    var_previ2 = var + '_previ2'
    df[var_previ1] = df.sort_values(by=['data'], axis=0)[var].shift(periods=-1)
    df[var_previ1].fillna(method='ffill', inplace=True)
    df[var_previ2] = df.sort_values(by=['data'], axis=0)[var_previ1].shift(periods=-1)
    df[var_previ2].fillna(method='ffill', inplace=True)
    return df

def split_train_test(estacio_df, train_percent):
    min_date = estacio_df.data.min()
    max_date = estacio_df.data.max()
    time_between = max_date - min_date
    train_cutoff = min_date + train_percent*time_between
    train = estacio_df[estacio_df.data <= train_cutoff]
    test = estacio_df[estacio_df.data > train_cutoff]
    return train, test