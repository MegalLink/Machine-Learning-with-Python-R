# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:17:18 2023

@author: jefer
"""

import pandas as pd 
from sklearn import preprocessing 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.model_selection import train_test_split


#preprocesado
dataset = pd.read_csv("../datasets/regression/50_Startups.csv")
x = dataset.iloc[:,:-1].values #important so it returns an 2d array
y = dataset.iloc[:,4].values
#Dataset ganancia de la empresa en base al gasto en r&d, administracion , marketing, dependiendo tamnbien del lugar(state)
# en este caso si hacemos un dummy del state a 0 y 1 1 para new york 0 para california
# la variable independiente ya contiene la informacion de california desde el inicio 
# cuando activamos para new york con 1 se suma el valor de california mas la diferencia de new york
# y =b0 + b1x1 + b2x2.......  b0 = independiente
# si agregamos las 2 dummy de newyork y california tenemos el problema de multicolinealidad , 
# dependencia circular entre las 2 variables, las 2 variables predicen del mismo modo
# si tuviera 3 , a,b,c dejaria solo 2 columnas para evitar esto

# transform categoric data
label_encoder = preprocessing.LabelEncoder()
x[:,3] = label_encoder.fit_transform(x[:,3]) #column to numbers
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],   
    remainder='passthrough'                        
)# transforma a dummy la columna
x = np.array(ct.fit_transform(x), dtype=np.float)#agregar las dummy
# evitar trampa de las variables dummy debemos eliminar una de las 3 generadas
x= x[:,1:]
# entrenar
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)
#modelo de regresion lineal multiple, igual que la simple solo que se le pasa muchas columnas xD
from sklearn.linear_model import LinearRegression
regression= LinearRegression()
regression.fit(x_train, y_train)

# Prediccion de los resultados
y_pred=regression.predict(x_test)

# Metodo eliminacion hacia atras, para eliminar variables que no sean estadisticamente significaticas
import statsmodels.api as sm
#agregamos columna que nos servira como nivel estadistico
#x= np.append(arr=x, values= np.ones((3, 4), dtype=int),axis=1) #agrega al ultimo
x=np.append(arr=np.ones((50, 1), dtype=int), values= x,axis=1)#agrega al principio la columna
sl=0.05
#Priemra iteracion con todas
x_opt= x[:,[0,1,2,3,4,5]]
regression_ols= sm.OLS(endog=y,exog=x_opt).fit() #este metodo necesita la columna de unos
regression_ols.summary()
#segunda iteracion solo quitamos la columna que no queremos y ajustamos denuevo el modelo
x_opt= x[:,[0,1,3,4,5]]
regression_ols= sm.OLS(endog=y,exog=x_opt).fit() #este metodo necesita la columna de unos
regression_ols.summary()

