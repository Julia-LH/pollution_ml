import pandas as pd
import os.path as path
from sklearn.metrics import r2_score

from classes import Polucio, Meteo
from funcions import loop_fill_split, pipeline_processat_train, pipeline_processat_test
from config import pol_arxiu, data_inici, area_geo, nom_area, contaminant, meteo_arxiu, var_arxiu, est_arxiu, pol_csv, meteo_csv, var_csv, est_csv, csv_path, models_params


# if __name__ == '__main__':


polucio = Polucio(pol_arxiu=pol_arxiu , data_inici=data_inici, area_geo=area_geo, nom_area=nom_area, contaminant=contaminant)
polucio_df = polucio.transformar_pol_arxiu()
contaminant = polucio.obtenir_nom_contaminant()

meteo = Meteo(meteo_arxiu=meteo_arxiu, var_arxiu=var_arxiu, est_arxiu=est_arxiu)
meteo_df = meteo.transformar_meteo_arxiu()

var_list = list(meteo.obtenir_noms_variables().values())
var_list.append(contaminant)

df = polucio_df.merge(meteo_df, how='outer', left_on=['data', 'nom_estacio'], right_on=['data_lectura', 'nom_estacio'])
df.drop(['data_lectura', 'codi_estacio', 'codi_meteo'], axis=1, inplace=True)

train_df, test_df = loop_fill_split(df=df, var_list=var_list)
x_train, y_train, enc, scaler, best_score, best_estimator = pipeline_processat_train(data=train_df, var_list=var_list, model_params=models_params)
r2score = pipeline_processat_test(data=test_df, var_list=var_list, encoder=enc, scaler=scaler, best_estimator=best_estimator, r2_score=r2_score)
print(best_estimator)
print(best_score)
print(r2score)