
from typing import Literal
from pydantic import BaseModel, Field
from datetime import date



# ===================== Entrada =====================
class ClienteInput(BaseModel):
    antiguedad_meses: int = Field(..., ge=0, le=60, description="Antigüedad del cliente en meses (1-60)")
    gasto_mensual: float = Field(..., gt=0, le=100, description="Gasto mensual del cliente")
    visitas_ultimo_mes: int = Field(..., ge=0, le=50, description="Número de visitas en el último mes")
    dias_desde_ultima_visita: int = Field(..., ge=0, le=120, description="Días desde la última visita")
    tickets_soporte: int = Field(..., ge=0, le=10, description="Número de tickets de soporte")
    plan: Literal["basico", "estandar", "premium"] = Field(..., description="Plan del cliente")
    metodo_pago: Literal["tarjeta", "transferencia", "efectivo"] = Field(..., description="Método de pago")
    descuento_activo: int = Field(..., ge=0, le=1, description="1 si tiene descuento activo, 0 si no")


# ===================== Salida =====================
class ClienteOutput(BaseModel):
    cancelo_predicho: int = Field(..., description="1 = cancela, 0 = sigue activo")
    probabilidad: float = Field(..., description="Probabilidad de que el cliente cancele")
    cancelacion: str = Field(..., description="Interpretación de la predicción")