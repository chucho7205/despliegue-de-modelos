from contextlib import asynccontextmanager   #funcion asincronica Conectar y desconectar bases de datos de forma asíncrona.
from typing import Literal  #cuando escribimos valores literales se usa en valores de familia comopresente y ausente

import joblib   #para cargar la libreria del bundle
import pandas as pd   #se van a hacer las mismas tranformaciones que se hicieron en la prueba
from fastapi import FastAPI, HTTPException   # clase FastAPI que nos permite construir la API y HTTPException para hacer solicitudes al servidor y nos devuelve algo
#tambien para saber que error puede dar
from pydantic import BaseModel, Field   #es una forma de poder modelar los datos como debemos modelar las entradas y las salidas
#las entradas son como la edad, alcohol, familia, etc y la salida es la prediccion, probabilidad
#field para indicar los campos que vamos a construir y nosotros podemos describir el campo


#antes de cargar la app cargar el bundle
NOMBRE_BUNDLE = r'modelo_demanda.joblib'   #copiar ruta de acceso relativa

estado_de_servicio = {'bundle': None}     #se crea la variable en diccionario que va a estar vacio

@asynccontextmanager    #decorador es una forma de poder hacerle un ciclo de vida, para cargar el modelo
async def lifespan (app:FastAPI):     #funcion asincronica

    estado_de_servicio['bundle'] = joblib.load(NOMBRE_BUNDLE)    #para poder cargar el bundle con joblib.load
    print ('bundle cargado correctamente')
    yield     #se usa in ciclos o funciones, se retorna, pausa y se reanuda el estado es como el return
    estado_de_servicio["bundle"] = None   #aqui se reinicia



#se construye la aplicacion, para inicializarla

app = FastAPI(title= "API Predicción de Enfermedad Cardiaca", 
              description = "Recibe datos clinicos de un paciente y predice el riego de cardiopatia coronoria (chd).", 
              version="1.0.0",
              lifespan=lifespan    #ejecuta el codigo antes que una aplicacion inicie
              )

#se va a crear una clase, la clase es la forma de acceder a los atributos y le indicamos a fastapi como debemos entregar el json
#field permite colocar los valores que podemos seleccionar de esa columna
# como es un numero variable se colocan 3 puntos eso representa que vamos apasar los valores y vamos a agregar una descripcion
#literal los valores de texto se los puedo pasar como en lista de definiciones

#todas las entradas que necesitamos para poder tener la prediccion
class PacienteInput(BaseModel):       #con basemodel se hacen las validaciones
    sbp: int = Field(...,description="Presión arterial sistólica"),
    Tabaco: float = Field(...,description="Tabaco acumulado (kg)"),
    ldl: float = Field(...,description="Colesterol LDL"),
    Adiposidad: float = Field(...,description="Adiposidad"),
    Familia: Literal['Presente',
                     'Ausente'] = Field(
                         ...,description="Antecendentes familiares de enfermedad cardíaca"),
    Tipo: int = Field(...,description="Comportamiento tipo-A"),
    Obesidad: float = Field(...,description="Obesidad"),
    Alcohol:float = Field(...,description="Consumo actual de alcohol"),
    Edad:int = Field(...,description="Edad")

#clase para las salidas que va a dar la prediccion
class PacienteOutput(BaseModel):   #basemodel nos permite hacer validaciones que la entrada o la salida sea del formato que se le dice
    chd_predicho: int     #va a dar  1 o 0 ya que es clasificacion (no se separa por comas)
    probabilidad: float
    riesgo: str    #interpretacion del 0 o 1

#un endpoint es para que dos sistemaso aplicaciones se comuniquen
@app.get('/')    #endpoint se pone ('/')para que cuando se le ponga el enlace te dirija al sitio
def estado ():    #es para ver si el estado esta cargado
    return{      #va a ser un diccionario
        'servicio': 'API de prediccion cardiaca',
        'modelo_cargado': estado_de_servicio['bundle'] is not None
    }



#predecir #/predecir es la ruta donde se va a mandar la info response model es el modelo de respuesta 
@app.post('/predecir', response_model=PacienteOutput)   #nuestro enlace /predecir y la respuesta response pacienteoutput es el modelo
def predecir (paciente: PacienteInput):   #parametro son todos los datos que necesitamospara predecir, estos datos tienen una estructura pydantic

    #validar modelo (mismo codigo que se uso para la prediccion)
    bundle = estado_de_servicio["bundle"]

    if bundle is None:       #validar si el bundle es vacion o no vacio
        raise HTTPException (status_code=503, detail='El modelo no esta cargado')   #raise palabra clave que permite poner un error a partir de una validacion

    fila = paciente.model_dump()

    fila["Familia"] = bundle["mapeo_familia"][fila["Familia"]]  #funcion del bundle
    X_nuevo = pd.DataFrame([fila])[bundle["columns"]]                                                        
    prediccion = bundle["pipeline"].predict(X_nuevo)[0] 
    probabilidad = bundle["pipeline"].predict_proba(X_nuevo)[0,1] 

#ahora vamos a retornar la estructura de paciente output
    return PacienteOutput (
        chd_predicho= prediccion,
        probabilidad= round(probabilidad,4),
        riesgo= 'alto' if prediccion == 1 else 'bajo'
    )