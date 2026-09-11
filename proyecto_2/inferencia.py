import pandas as pd
import numpy as np
from features import crear_features  #nos traemos la funcion que creamos del archivo features la funcion crear_features



#funcion pronosticar por dia
#en el bundle se guarda el modelo, las columnas y la configuracion minima de historia requerida para poder pronosticar
def pronosticar(bundle, historial, horizonte=14):
    """
    bundle    : dict con modelo, columnas y configuracion
    historial : DataFrame con columnas date, store, item, sales (al menos 28 filas, sin huecos)
    horizonte : cantidad de dias a pronosticar
    devuelve  : lista de dicts {fecha, prediccion}
    """
    modelo, columnas = bundle["modelo"], bundle["columnas"]
    minimo = bundle["min_historial"]

    h = historial.copy()   #copia del historial de validacion
    h["date"] = pd.to_datetime(h["date"])   
    h = h.sort_values("date").reset_index(drop=True)

    if len(h) < minimo:
        raise ValueError(f"Se requieren al menos {minimo} dias de historia, se recibieron {len(h)}")   #se realiza la validacion con minimo de 1 dia
    if h.date.diff().dropna().ne(pd.Timedelta(days=1)).any():   #si esto falla, es decir, si hay algun dia que no sea consecutivo, se lanza un error
        raise ValueError("El historial tiene fechas faltantes o desordenadas")  # que marque error

    store, item = int(h.store.iloc[-1]), int(h.item.iloc[-1])  #seleccionamos las tiendas y e item
    resultado = []

    for _ in range(horizonte):  #hacemos un for para poder tener el resultado
        siguiente = h.date.iloc[-1] + pd.Timedelta(days=1)  #aqui se va aumentando el dia
        h = pd.concat(    #concatenamos la informacion resultante con la fecha, la tienda, el item y las ventas que son nulas porque es lo que queremos predecir
            [h, pd.DataFrame([{"date": siguiente, "store": store, "item": item, "sales": np.nan}])],
            ignore_index=True,
        )
        fila = crear_features(h).iloc[[-1]]   #rellenamos la fila correspondiente
        pred = max(0.0, float(modelo.predict(fila[columnas])[0]))   #hacemos la prediccion y nos aseguramos que no sea negativa
        h.loc[h.index[-1], "sales"] = pred          # se realimenta como si fuera dato real, se coloca en la columna sales el valor predicho para que en la siguiente iteracion se pueda usar como dato real y asi sucesivamente
        resultado.append({"fecha": siguiente.date().isoformat(), "prediccion": round(pred, 2)})   #y como resultado tendra la fecha y la prediccion

    return resultado