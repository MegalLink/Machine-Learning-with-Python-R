# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:49:45 2023
Pre procesamiento
@author: jefer
"""
import numpy as np # operaciones matematicas
import matplotlib.pyplot as plt # dibujar
import pandas as pd #carga y manipulacion de datos desde archivos

dataset = pd.read_csv("../datasets/preprocessing/dirty_data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values