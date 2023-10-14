from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor


csv_path = 'datasets'
#polucio
pol_csv = 'qualitat_aire.csv'
pol_arxiu = 'datasets/qualitat_aire.csv'
data_inici = '2021-01-01'
area_geo = 'municipi'
nom_area = 'Barcelona'
contaminant = 'NO'
#meteo
meteo_csv = 'meteo_test8.csv'
meteo_arxiu = 'datasets/meteo_test8.csv'
var_arxiu = 'datasets/variables_meteo_reduit.csv'
var_csv = 'variables_meteo_reduit.csv'
est_arxiu = 'datasets/estacions_meteo_polucio.csv'
est_csv = 'estacions_meteo_polucio.csv'
#model
training_set='datasets/train_df.csv'



models_params = {'LinearRegression': {'name' : LinearRegression(), 'params' : {'n_jobs': [1, 2, 3, 4, 5]}}, 
                'XGBRegressor' : {'name':XGBRegressor(), 'params' : {'max_depth':[1, 2, 3, 4, 5, 6], 'eta':[0.1, 0.01, 0.001, 0.0001]}},
                'RandomForestRegressor': {'name' : RandomForestRegressor(), 'params' : {'max_depth':[1, 2, 3, 4, 5, 6], 'n_jobs': [1, 2, 3, 4, 5]}},
                'Ridge' : {'name' : Ridge(), 'params' : {'max_iter':[1, 2, 3, 4, 5, 6], 'alpha':[0.1, 0.01, 0.001, 0.0001]}}}
