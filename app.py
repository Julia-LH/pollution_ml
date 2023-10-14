from flask import Flask
from joblib import load

# from scripts.preprocessat import best_estimator, best_score
# from scripts.funcions import 
from scripts.config import models_params

encoder = load('datasets/encoder.joblib')
scaler = load('datasets/scaler.joblib')
estimator = load('datasets/best_estimator.joblib')

print(models_params.keys())

# print(best_estimator, best_score)

# migrar dict con modelos a config e importarlo en funciones y directamente como lista de dict.keys en app.py
# importar elementos del modelo y hacer la prediccion en la web
# escribir los inputs de variables meteo en la web