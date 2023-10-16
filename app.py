from flask import Flask, jsonify

from scripts.config import models_list, est_arxiu, contaminant, nom_area
from server.app_objects import estacions, nom_estació

app = Flask(__name__)
app.config['DEBUG'] = True


@app.route('/')
@app.route('/home')
def home():
    text = '''
    <h1>Predicció de polució de {}</h1>
    <h2>{}</h2>
    '''.format(contaminant, nom_area)
    return text, jsonify(models_list)


@app.route('/prueba1')
def param_list():
    return jsonify(models_list)


@app.route('/prueba2')
def prueba2():
    if nom_estació:
        df = estacions.loc[estacions['nom_estacio']==nom_estació]
        return df.to_string()
    else:
        return "Nom d'estació no indicat"


# @app.route('/prueba3', methods=['GET', 'POST'])
# def prueba():
#     # flask_form
#     return 


# migrar dict con modelos a config e importarlo en funciones y directamente como lista de dict.keys en app.py
# importar elementos del modelo y hacer la prediccion en la web
# escribir los inputs de variables meteo en la web