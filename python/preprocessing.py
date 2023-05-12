# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:49:45 2023
Pre procesamiento
@author: jefer
"""
import numpy as np # operaciones matematicas
import matplotlib.pyplot as plt # dibujar
import pandas as pd #carga y manipulacion de datos desde archivos
from sklearn.impute import SimpleImputer # sintaxis para importar solo una parte(funcion) de la libreria
from sklearn import preprocessing #transformar datos categoricos
from sklearn.preprocessing import OneHotEncoder #Crear varibles dummy para los datos categoricos
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("../datasets/preprocessing/dirty_data.csv")
independent_values = dataset.iloc[:,:-1].values
dependant_values = dataset.iloc[:,3].values
# 1) Tratar datos dañados
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean") #valor desconocido, estrategia, media de fila=1 o columna=0
imputer = imputer.fit(independent_values[:, 1:3]) #escogemos la columna age y salary en python la syntaxis excluye el valor de la derecha
independent_values[:, 1:3] = imputer.transform(independent_values[:, 1:3])

# 2) codificar datos categoricos ejemplo los paises darles un numero a cada uno
# para que parezcan ordinales(tienen cierto orden) en el caso de los paises no importa el orden
label_encoder_country = preprocessing.LabelEncoder()
independent_values[:,0] = label_encoder_country.fit_transform(independent_values[:,0])

# en vez de tener una sola columna con cada pais asignado su key creamos una columna por cada tipo de país
# para que no se confunda el modelo y piense que tiene relevancia el valor de los paises codificados
# cuando se tiene solo 2 valores no es necesario
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   
    remainder='passthrough'                        
)
independent_values = np.array(ct.fit_transform(independent_values), dtype=np.float)

# para la columna Retired  no es necesario por que solo tiene 2 valores true o false
label_encoder_retired = preprocessing.LabelEncoder()
dependant_values = label_encoder_retired.fit_transform(dependant_values)

# 3) Dividir el dataset en entrenamiento 70 o 80 % del dataset y testing 30 o 20% del dataset
independant_values_train,independant_values_test,dependant_values_train,dependant_values_test = train_test_split(independent_values,dependant_values,test_size=0.2,random_state=0)
# el random state es la semilla para dividir de forma aleatoria 
# el overfiting es cuando da problemas el modelo de prediccion por testear datos que no conoce

# 4) Escalar los datos , si existe una variable que es mucha mas mayor en valor que las otras hace que 
# los valores de las otras sean insignificantes aplicanto la distancia entre dos puntos con x,y los valores de 2 variables
# Ejemplo el salario con la edad, lo mejor seria escalar la edad y el salario entre -1 y 1
# esto evita que unas variables sean mas importantes que otras
# tecnicas estandarizacion (campana de gaus) y normalizacion
#Estandarizacion = (x - media(x))/desviacionEstandar(x)
#Normalizacion = x-min(x)/((max(x)-min(x)))
scaler_independant= StandardScaler() # entre -1 y 1
independant_values_train = scaler_independant.fit_transform(independant_values_train)
independant_values_test = scaler_independant.transform(independant_values_test)# usamos solo transform 
# se debe escalar los datos dummy?, se puede hacer las 2 :v no hay ventaja en escalar las dummy
# el algoritmo converge mas rapidamente estandarizando asi que hay que hacerlo casi siempre aunque no se use
# algoritmos con distancias euclidianas





