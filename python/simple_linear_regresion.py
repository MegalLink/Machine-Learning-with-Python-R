# -*- coding: utf-8 -*-
"""
Created on Sun May 14 19:45:59 2023

@author: jefer
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#preprocesado
dataset = pd.read_csv("../datasets/regression/salary_data.csv")
x = dataset.iloc[:,:-1].values #important so it returns an 2d array
y = dataset.iloc[:,1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

#crear modelo de regresion lineal
regression= LinearRegression()

regression.fit(x_train,y_train)

#predecir
y_pred = regression.predict(x_test)

#visualizar los resultados
plt.scatter(x_train,y_train, color = "red")
plt.plot(x_train,regression.predict(x_train),color = "blue")
plt.title("Sueldo vs años de experiencia(Entrenamiento)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()


plt.scatter(x_test,y_test, color = "red")
plt.plot(x_train,regression.predict(x_train),color = "blue")
plt.title("Sueldo vs años de experiencia(Test)")
plt.xlabel("Años de experiencia")
plt.ylabel("Sueldo (en $)")
plt.show()