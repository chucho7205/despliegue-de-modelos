
from pydantic import BaseModel, Field
from datetime import date


class RegistroHistorico(BaseModel):
    fecha: date 
    unidades: float = Field(ge=0, description="Número de unidades vendidas un dia, no puede ser negativo")




class SolicitudPronostico(BaseModel):

    store: int = Field(ge=1, le=10, description="el numero de la tienda (1-10)")
    item: int = Field(ge=1, le=50, description="el numero del item (1-50)")
    
    historial: list[RegistroHistorico] = Field(min_length=28, max_length=365, 
                                               description="histico reciente de la serie, minimo 28 dias y maximo 365 dias")
    horizonte: int = Field(default=14, ge=1, le=28,description="Número de días a pronosticar de 1 a 28")