import pandas as pd

from classes import Polucio, Meteo
from funcions import omplir_buits, split_train_test
from config import pol_arxiu, data_inici, area_geo, nom_area, contaminant, meteo_arxiu, var_arxiu, est_arxiu


polucio = Polucio(pol_arxiu=pol_arxiu, data_inici=data_inici, area_geo=area_geo, nom_area=nom_area, contaminant=contaminant)
polucio_df = polucio.transformar_pol_arxiu()
contaminant = polucio.obtenir_nom_contaminant()

meteo = Meteo(meteo_arxiu=meteo_arxiu, var_arxiu=var_arxiu, est_arxiu=est_arxiu)
meteo_df = meteo.transformar_meteo_arxiu()
var_list = list(meteo.obtenir_noms_variables().values())
var_list.append(contaminant)
df= polucio_df.merge(meteo_df, how='outer', left_on=['data', 'nom_estacio'], right_on=['data_lectura', 'nom_estacio'])
df.drop(['data_lectura', 'codi_estacio', 'codi_meteo'], axis=1, inplace=True)
train_df = pd.DataFrame()
test_df = pd.DataFrame()
for estacio in set(df['nom_estacio']):
    estacio_df = df[df['nom_estacio']==estacio]
    for var in var_list:
        estacio_df = omplir_buits(df=estacio_df, var=var)

    train, test = split_train_test(estacio_df, 0.75)
    train_df = pd.concat([train_df, train], ignore_index=True)
    test_df = pd.concat([test_df, test], ignore_index=True)

#test_df.to_csv('datasets/test_df.csv', index=False)
#train_df.to_csv('datasets/train_df.csv', index=False)









    