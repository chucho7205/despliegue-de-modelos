# Guía completa: Despliegue de la API de Cancelación de Clientes

Documento con todos los pasos realizados para crear el bundle, los archivos `esquema.py` y `main.py`, levantar el servicio y probarlo con **Thunder Client** en Visual Studio Code.

---

## 1. Contexto del proyecto

- **Problema**: Predecir si un cliente va a cancelar su suscripción.
- **Modelo**: Regresión Logística dentro de un `Pipeline` de scikit-learn.
- **Dataset**: `clientes.csv` (6000 filas).
- **Variable objetivo**: `cancelo` (0 = sigue activo, 1 = cancela).

### Columnas de entrada (features)

| Columna                    | Tipo   | Valores / rango          |
|---------------------------|--------|---------------------------|
| `antiguedad_meses`        | int    | 1 – 60                    |
| `gasto_mensual`           | float  | 5.00 – 58.39              |
| `visitas_ultimo_mes`      | int    | 0 – 29                    |
| `dias_desde_ultima_visita`| int    | 0 – 90                    |
| `tickets_soporte`         | int    | 0 – 5                     |
| `plan`                    | str    | `basico`, `estandar`, `premium` |
| `metodo_pago`             | str    | `tarjeta`, `transferencia`, `efectivo` |
| `descuento_activo`        | int    | 0 o 1                     |

---

## 2. Parte 1 — Crear y guardar el bundle (en el notebook)

En el notebook `entrenar_modelo.ipynb`, al final de la sección **TAREA**:

### 2.1 Armar el diccionario `bundle`

```python
from datetime import datetime
import sklearn

bundle = {
    "pipeline": modelo,                    # Pipeline entrenado (preprocesamiento + modelo)
    "columns": list(X.columns),            # Orden de columnas que espera el modelo
    "metrica": {
        "f1": float(round(f1, 3)),
        "auc": float(round(auc, 3))
    },
    "metadata": {
        "fecha_entrenamiento": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "modelo": "Regresión Logística",
        "dataset": "clientes.csv",
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "joblib": joblib.__version__
    }
}
```

### 2.2 Guardar el bundle

```python
import joblib
joblib.dump(bundle, "modelo_churn.joblib")
```

### 2.3 Verificar que el bundle funciona

```python
bundle_cargado = joblib.load("modelo_churn.joblib")

def predecir_cancelacion(cliente_nuevo: dict, bundle: dict) -> dict:
    fila = dict(cliente_nuevo)
    X_nuevo = pd.DataFrame([fila])[bundle["columns"]]
    prediccion = bundle["pipeline"].predict(X_nuevo)[0]
    probabilidad = bundle["pipeline"].predict_proba(X_nuevo)[0, 1]
    return {
        "cancelo_predicho": int(prediccion),
        "probabilidad": round(float(probabilidad), 4)
    }

# Prueba
cliente_nuevo = X_prueba.iloc[0].to_dict()
resultado = predecir_cancelacion(cliente_nuevo, bundle_cargado)
print(resultado)
```

**Importante**: Copia el archivo `modelo_churn.joblib` a la carpeta del proyecto donde vas a poner la API (`proyecto_3`).

---

## 3. Parte 2 — Archivo `esquema.py`

Crea el archivo `esquema.py` (o pon los modelos Pydantic dentro de `main.py`).  
Define la **entrada** y la **salida** de la API:

```python
from typing import Literal
from pydantic import BaseModel, Field


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
```

### Notas del esquema
- Los valores de `plan` y `metodo_pago` van en **minúsculas** (igual que en el CSV).
- `descuento_activo` es `0` o `1` (entero), no `bool`.
- Los límites (`ge`, `le`, `gt`) evitan datos absurdos.

---

## 4. Parte 3 — Archivo `main.py`

```python
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# Ruta del bundle (AJUSTA ESTA RUTA A TU COMPUTADORA)
# ============================================================
NOMBRE_BUNDLE = r'C:\Users\chuch\Documents\espy\Despliegue y Producción de Modelos de Machine Learning\proyecto_3\modelo_churn.joblib'

estado_de_servicio = {'bundle': None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar la API y lo libera al cerrar."""
    try:
        estado_de_servicio['bundle'] = joblib.load(NOMBRE_BUNDLE)
        print('Bundle cargado correctamente')
    except Exception as e:
        print(f'Error al cargar el bundle: {e}')
        estado_de_servicio['bundle'] = None
    yield
    estado_de_servicio['bundle'] = None


app = FastAPI(
    title="API Predicción de cancelación de clientes",
    description="Recibe datos de un cliente y predice el riesgo de cancelación.",
    version="1.0.0",
    lifespan=lifespan
)


# ===================== Modelos de entrada y salida =====================
class ClienteInput(BaseModel):
    antiguedad_meses: int = Field(..., ge=0, le=60, description="Antigüedad del cliente en meses")
    gasto_mensual: float = Field(..., gt=0, le=100, description="Gasto mensual del cliente")
    visitas_ultimo_mes: int = Field(..., ge=0, le=50, description="Número de visitas en el último mes")
    dias_desde_ultima_visita: int = Field(..., ge=0, le=120, description="Días desde la última visita")
    tickets_soporte: int = Field(..., ge=0, le=10, description="Número de tickets de soporte")
    plan: Literal["basico", "estandar", "premium"] = Field(..., description="Plan del cliente")
    metodo_pago: Literal["tarjeta", "transferencia", "efectivo"] = Field(..., description="Método de pago")
    descuento_activo: int = Field(..., ge=0, le=1, description="1 si tiene descuento activo, 0 si no")


class ClienteOutput(BaseModel):
    cancelo_predicho: int = Field(..., description="1 = cancela, 0 = sigue activo")
    probabilidad: float = Field(..., description="Probabilidad de cancelar")
    cancelacion: str = Field(..., description="Interpretación de la predicción")


# ===================== Endpoints =====================
@app.get('/')
def estado():
    return {
        'servicio': 'API de predicción de cancelación de clientes',
        'modelo_cargado': estado_de_servicio['bundle'] is not None
    }


@app.post('/predecir', response_model=ClienteOutput)
def predecir(cliente: ClienteInput):
    bundle = estado_de_servicio.get("bundle")

    if bundle is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado")

    try:
        # Convertir input de Pydantic a diccionario
        fila = cliente.model_dump()          # (en versiones viejas: cliente.dict())

        # DataFrame con el orden de columnas que espera el modelo
        X_nuevo = pd.DataFrame([fila])[bundle["columns"]]

        # Predicción
        prediccion = int(bundle["pipeline"].predict(X_nuevo)[0])
        probabilidad = float(bundle["pipeline"].predict_proba(X_nuevo)[0, 1])

        return ClienteOutput(
            cancelo_predicho=prediccion,
            probabilidad=round(probabilidad, 4),
            cancelacion="cancela" if probabilidad >= 0.5 else "sigue activo"
        )

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Falta una columna requerida: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al realizar la predicción: {str(e)}")


# Para ejecutar localmente
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
```

### Puntos clave del `main.py`
1. El bundle se carga **una sola vez** al arrancar (lifespan).
2. Se usa `cliente.model_dump()` (o `.dict()` en Pydantic v1).
3. Se respeta el orden de columnas con `bundle["columns"]`.
4. Se fuerza que la probabilidad salga redondeada y se interpreta el resultado.

---

## 5. Parte 4 — Levantar el servicio

### 5.1 Activar el entorno virtual

```bash
# Desde la carpeta del proyecto
.venv\Scripts\activate          # Windows
# o
source .venv/bin/activate       # Linux / Mac
```

### 5.2 Instalar dependencias (si faltan)

```bash
pip install fastapi uvicorn joblib pandas scikit-learn pydantic
```

### 5.3 Ejecutar la API

**Opción A** — Estando en la carpeta donde está `main.py`:

```bash
uvicorn main:app --reload
```

**Opción B** — Desde la carpeta padre (si el módulo se llama `proyecto_3`):

```bash
uvicorn proyecto_3.main:app --reload
```

Debes ver algo como:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
Bundle cargado correctamente
```

### 5.4 Probar que está viva

Abre el navegador en:

- http://127.0.0.1:8000/          → estado del servicio
- http://127.0.0.1:8000/docs      → documentación interactiva (Swagger)

---

## 6. Parte 5 — Probar con Thunder Client (VS Code)

### 6.1 Instalar la extensión
1. Abre Visual Studio Code.
2. Ve a Extensiones (`Ctrl+Shift+X`).
3. Busca **Thunder Client** e instálala.

### 6.2 Crear una petición GET (estado)

1. Abre Thunder Client (icono del rayo en la barra lateral).
2. New Request.
3. Método: **GET**
4. URL: `http://127.0.0.1:8000/`
5. Send.

**Respuesta esperada:**
```json
{
  "servicio": "API de predicción de cancelación de clientes",
  "modelo_cargado": true
}
```

### 6.3 Crear una petición POST (predicción)

1. New Request.
2. Método: **POST**
3. URL: `http://127.0.0.1:8000/predecir`
4. Pestaña **Body** → selecciona **JSON**.
5. Pega el siguiente cuerpo:

```json
{
  "antiguedad_meses": 10,
  "gasto_mensual": 14.0,
  "visitas_ultimo_mes": 4,
  "dias_desde_ultima_visita": 26,
  "tickets_soporte": 1,
  "plan": "basico",
  "metodo_pago": "efectivo",
  "descuento_activo": 0
}
```

6. Send.

**Respuesta esperada (ejemplo):**
```json
{
  "cancelo_predicho": 1,
  "probabilidad": 0.881,
  "cancelacion": "cancela"
}
```

### 6.4 Otro ejemplo (cliente fiel)

```json
{
  "antiguedad_meses": 20,
  "gasto_mensual": 45.0,
  "visitas_ultimo_mes": 8,
  "dias_desde_ultima_visita": 10,
  "tickets_soporte": 0,
  "plan": "premium",
  "metodo_pago": "tarjeta",
  "descuento_activo": 0
}
```

Debería devolver una probabilidad baja y `"cancelacion": "sigue activo"`.

---

## 7. Errores comunes y cómo resolverlos

| Error | Causa | Solución |
|-------|-------|----------|
| `Import string "xxx" must be in format "<module>:<attribute>"` | Falta el `:` en el comando uvicorn | Usa `uvicorn main:app --reload` |
| `modelo_cargado: false` | Ruta del `.joblib` incorrecta | Revisa `NOMBRE_BUNDLE` |
| `422 Unprocessable Entity` | Valor inválido en el JSON (ej. plan en mayúsculas) | Usa exactamente `basico`, `estandar`, `premium` |
| `KeyError` / columna faltante | El bundle no tiene la clave `"columns"` o el orden es distinto | Vuelve a generar el bundle |
| `ModuleNotFoundError: No module named 'joblib'` | Entorno virtual sin dependencias | `pip install joblib pandas scikit-learn fastapi uvicorn` |
| Predicción diferente a la del notebook | Estás usando otro archivo `.joblib` o columnas distintas | Confirma que es el mismo `modelo_churn.joblib` |

---

## 8. Estructura final recomendada del proyecto

```
proyecto_3/
├── modelo_churn.joblib          # Bundle guardado
├── main.py                      # API FastAPI
├── esquema.py                   # (opcional) modelos Pydantic separados
├── clientes.csv                 # Dataset original
├── entrenar_modelo.ipynb        # Notebook de entrenamiento
└── requirements.txt             # Dependencias
```

### Ejemplo de `requirements.txt`

```
fastapi
uvicorn
joblib
pandas
scikit-learn
pydantic
```

---

## 9. Resumen del flujo completo

1. Entrenar el modelo en el notebook.
2. Armar el `bundle` con `pipeline`, `columns`, métricas y metadata.
3. Guardar con `joblib.dump(..., "modelo_churn.joblib")`.
4. Verificar carga + predicción en el notebook.
5. Crear `esquema.py` (o definir las clases en `main.py`).
6. Crear `main.py` con lifespan, endpoints `/` y `/predecir`.
7. Activar el entorno virtual y ejecutar `uvicorn main:app --reload`.
8. Probar con Thunder Client (GET `/` y POST `/predecir`).

Con esto el servicio queda desplegado localmente y listo para consumir.
