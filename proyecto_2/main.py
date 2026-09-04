from contextlib import asynccontextmanager   #funcion asincronica Conectar y desconectar bases de datos de forma asíncrona.
from typing import Literal  #cuando escribimos valores literales se usa en valores de familia comopresente y ausente

import joblib   #para cargar la libreria del bundle
import pandas as pd   #se van a hacer las mismas tranformaciones que se hicieron en la prueba
from fastapi import FastAPI, HTTPException   
from pydantic import BaseModel, Field   
from proyecto_2.inferencia import pronosticar
from proyecto_2.esquema import  SolicitudPronostico

#antes de cargar la app cargar el bundle
NOMBRE_BUNDLE = r'C:\Users\chuch\Documents\espy\Despliegue y Producción de Modelos de Machine Learning\proyecto_2\modelo_demanda.joblib'   

estado_de_servicio = {'bundle': None}     #se crea la variable en diccionario que va a estar vacio

@asynccontextmanager    #decorador es una forma de poder hacerle un ciclo de vida, para cargar el modelo
async def lifespan (app:FastAPI):     #funcion asincronica

    estado_de_servicio['bundle'] = joblib.load(NOMBRE_BUNDLE)    #para poder cargar el bundle con joblib.load
    print ('bundle cargado correctamente')
    yield     #se usa in ciclos o funciones, se retorna, pausa y se reanuda el estado es como el return
    estado_de_servicio["bundle"] = None   #aqui se reinicia



#se construye la aplicacion, para inicializarla

app = FastAPI(title= "API de pronostico de demanda", 
              description = "pronostico de demanda por forecast", 
              version="1.0.0",
              lifespan=lifespan    #ejecuta el codigo antes que una aplicacion inicie
              )

@app.get('/')    #endpoint se pone ('/')para que cuando se le ponga el enlace te dirija al sitio
def estado ():    #es para ver si el estado esta cargado
    return{      #va a ser un diccionario
        'servicio': 'API de pronostico de demanda',
        'modelo_cargado': estado_de_servicio['bundle'] is not None
    }


#predecir
@app.post('/predecir')  
def predecir (datos: SolicitudPronostico):  #va a ser todo en diccionario

    store = datos.store
    item = datos.item
    horizonte = datos.horizonte
    registros = datos.historial    #requiere un historial minimo de 28 dias, viene en json

    

    historial = pd.DataFrame(  #se convierte a dataframe
        {                       #se haceun diccionario
        'date': pd.to_datetime([r.fecha for r in registros]),    #lista de comprension para tomar los valores y pasarlos a fechas
        'store': store,
        'item': item,
        'sales': [r.unidades for r in registros]     #lista de comprension para 
    }
    )

    bundle = estado_de_servicio['bundle']

    pronostico = pronosticar(bundle,historial,horizonte)   #funcion pronosticar que de parametros va a tener

    #pronosticar devuelve una lista con la fecha y la prediccion

    return{
        'store': store,
        'item': item,
        'pronostico': pronostico
    }