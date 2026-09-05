from contextlib import asynccontextmanager   #funcion asincronica Conectar y desconectar bases de datos de forma asíncrona.
from typing import Literal  #cuando escribimos valores literales se usa en valores de familia comopresente y ausente

import joblib   #para cargar la libreria del bundle
import pandas as pd   #se van a hacer las mismas tranformaciones que se hicieron en la prueba
from fastapi import FastAPI, HTTPException   # clase FastAPI que nos permite construir la API y HTTPException para hacer solicitudes al servidor y nos devuelve algo
from pydantic import BaseModel, Field   #es una forma de poder modelar los datos como debemos modelar las entradas y las salidas



#antes de cargar la app cargar el bundle
NOMBRE_BUNDLE = r'C:\Users\chuch\Documents\espy\Despliegue y Producción de Modelos de Machine Learning\proyecto_3\modelo_churn.joblib'  

estado_de_servicio = {'bundle': None}     #se crea la variable en diccionario que va a estar vacio

@asynccontextmanager    #decorador es una forma de poder hacerle un ciclo de vida, para cargar el modelo
async def lifespan (app:FastAPI):     #funcion asincronica

    estado_de_servicio['bundle'] = joblib.load(NOMBRE_BUNDLE)    #para poder cargar el bundle con joblib.load
    print ('bundle cargado correctamente')
    yield     #se usa in ciclos o funciones, se retorna, pausa y se reanuda el estado es como el return
    estado_de_servicio["bundle"] = None   #aqui se reinicia



#se construye la aplicacion, para inicializarla

app = FastAPI(title= "API Predicción de cancelacion de clientes", 
              description = "Recibe datos clinicos de un cliente  y predice el riego de cancelacion.", 
              version="1.0.0",
              lifespan=lifespan    #ejecuta el codigo antes que una aplicacion inicie
              )


class ClienteInput(BaseModel):
    antiguedad_meses: int = Field(...,description="Antiguedad del cliente en meses")
    gasto_mensual: float = Field(...,description="Gasto mensual del cliente")
    visitas_ultimo_mes: int = Field(...,description="Número de visitas en el último mes")
    dias_desde_ultima_visita: int = Field(...,description="Días desde la última visita")
    tickets_soporte: int = Field(...,description="Número de tickets de soporte")
    plan: Literal["basico", "estandar", "premium"] = Field(...,description="Plan del cliente")
    metodo_pago: Literal["tarjeta", "transferencia", "efectivo"] = Field(...,description="Método de pago del cliente")
    descuento_activo: int = Field(...,ge=0, le=1, description="1 si tiene descuento activo, 0 si no")


class ClienteOutput(BaseModel):
    cancelo_predicho: int = Field(..., description="1 = cancela, 0 = sigue activo")
    probabilidad: float = Field(..., description="Probabilidad de cancelar")
    cancelacion: str = Field(..., description="Interpretación de la predicción")

@app.get('/')    
def estado ():    #es para ver si el estado esta cargado
    return{      #va a ser un diccionario
        'servicio': 'API de prediccion de cancelacion de clientes',
        'modelo_cargado': estado_de_servicio['bundle'] is not None
    }


@app.post('/predecir', response_model=ClienteOutput)   #nuestro enlace /predecir y la respuesta response clienteoutput es el modelo
def predecir (cliente: ClienteInput):   #parametro son todos los datos que necesitamospara predecir, estos datos tienen una estructura pydantic

    #validar modelo (mismo codigo que se uso para la prediccion)
    bundle = estado_de_servicio["bundle"]

    if bundle is None:       #validar si el bundle es vacion o no vacio
        raise HTTPException (status_code=503, detail='El modelo no esta cargado') 

    fila = dict(cliente)   
    X_nuevo = pd.DataFrame([fila])[bundle["columns"]]  
    prediccion = bundle["pipeline"].predict(X_nuevo)[0]  
    probabilidad = bundle["pipeline"].predict_proba(X_nuevo)[0,1] 


    return ClienteOutput(
        cancelo_predicho=prediccion, 
        probabilidad =round(probabilidad, 4),
        cancelacion= 'cancela' if probabilidad >= 0.5  else 'sigue activo'
        )

  

# Para ejecutar localmente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)