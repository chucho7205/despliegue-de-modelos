import requests     #libreria para hacer peticiones en la web

url = 'http://127.0.0.1:8000'   #sin el slash

r = requests.get('http://127.0.0.1:8000/')   #la r significa response 
#  r=requests.get(f'{url}/')   #otro metodo que es: se concatena la variable url de la direccion mas el slash y la f para insrtar variables
print(r.json())

paciente = {    #se crea una variable y se pasa el diccionario que esta requests
  "sbp": 160,
  "Tabaco": 12,
  "ldl": 5.73,
  "Adiposidad": 23.11,
  "Familia": "Presente",
  "Tipo": 49,
  "Obesidad": 25.3,
  "Alcohol": 97.2,
  "Edad": 52
}

r = requests.post('http://127.0.0.1:8000/predecir',json=paciente)   #a la ruta se le agrega el endpoint /predecir y el post permite agregar en jason
print(r.json())
                  