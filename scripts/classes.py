import pandas as pd
from datetime import datetime

class Polucio:

    def __init__(self, arxiu, data_inici, area_geo, nom_area, contaminant) -> None:
        self.arxiu = arxiu
        self.data_inici = data_inici
        self.area_geo = area_geo
        self.nom_area = nom_area    
        self.contaminant = contaminant
        self.data = 'DATA'
        self.format_data = '%d/%m/%Y'
        self.columns = ['NOM ESTACIO', 'DATA','CONTAMINANT', 'TIPUS ESTACIO', 'AREA URBANA', 'MUNICIPI',
                        'NOM COMARCA', '01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
                        '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
                        '19h', '20h', '21h', '22h', '23h', '24h', 'ALTITUD', 'LATITUD',
                        'LONGITUD']
        self.variables_agrupacio = ['CONTAMINANT', 'MUNICIPI', 'NOM COMARCA']
        self.mesures_hora = ['01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
                             '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
                             '19h', '20h', '21h', '22h', '23h', '24h'] 

    
    def obrir_arxiu(self):
        date_parser = lambda x:datetime.strptime(x, self.format_data)
        df = pd.read_csv(self.arxiu, usecols=self.columns, parse_dates=[self.data], date_parser=date_parser)
        df.rename(columns=lambda x: x.replace(' ', '_').lower(), inplace=True)
        df = df[(df[self.area_geo]==self.nom_area) & (df[self.data.replace(' ', '_').lower()]>self.data_inici) & (df['contaminant']==self.contaminant)]
        return df.reset_index(drop=True)

    def transformar_arxiu(self):
        df = self.obrir_arxiu()
        mesures_hora = [x.replace(' ', '_').lower() for x in self.mesures_hora]
        variables_agrupacio = [x.replace(' ', '_').lower() for x in self.variables_agrupacio]
        nom_contaminant = self.contaminant + '_dia'
        df[nom_contaminant] = df[mesures_hora].mean(axis=1)
        return df.drop(variables_agrupacio + mesures_hora, axis=1)
    


class Variables:

    def __init__(self, arxiu):
        self.arxiu = arxiu
        self.columns = ['codi_variable', 'nom_variable']
        self.dtypes={'codi_variable':'str'}

    def obtenir_noms_variables(self):
        df = pd.read_csv(self.arxiu, usecols=self.columns, dtype=self.dtypes)
        return df.set_index('codi_variable')['nom_variable'].to_dict()

    

class Meteo:

    def __init__(self, meteo_arxiu, var_arxiu, est_arxiu) -> None:
        self.arxiu = meteo_arxiu
        self.data ='data_lectura'
        self.format_data = '%Y-%m-%d'
        self.columns = ['codi_estacio', 'codi_variable', 'data_lectura', 'valor_lectura']
        self.dtypes = {'codi_variable':'str'}
        self.columns_grup = ['data_lectura', 'codi_variable', 'codi_estacio'] #
        self.pivot_column = 'codi_variable'
        self.var_arxiu = var_arxiu
        self.valor_variables = 'valor_lectura'
        self.est_arxiu = est_arxiu
    

    def obrir_arxiu(self):
        date_parser = lambda x:datetime.strptime(x, self.format_data)
        df = pd.read_csv(self.arxiu, usecols=self.columns, dtype=self.dtypes, parse_dates=[self.data], date_parser=date_parser)
        return df.rename(columns=lambda x: x.replace(' ', '_').lower())
    

    def transformar_arxiu(self):
        df = self.obrir_arxiu()
        variables = Variables(self.var_arxiu)
        nom_variables = variables.obtenir_noms_variables()
        estacions = pd.read_csv(self.est_arxiu, usecols=['codi_meteo', 'nom_estacio'])
        df = df.groupby(self.columns_grup)[self.valor_variables].mean().unstack(self.pivot_column).reset_index()
        df.rename(nom_variables, axis='columns', inplace=True)
        return df.merge(estacions, left_on='codi_estacio', right_on='codi_meteo')
        







if __name__ == '__main__':
    meteo = Meteo(meteo_arxiu='meteo_dataset\meteo_test6.csv', var_arxiu='meteo_dataset/variables_meteo_reduit.csv', est_arxiu='meteo_dataset\estacions_meteo_polucio.csv')
    test = meteo.transformar_arxiu()
    print(test.head())











