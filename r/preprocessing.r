# Pre processing

# 1) tratar datos dañados
dataset = read.csv('./datasets/preprocessing/dirty_data.csv')
dataset$Age = ifelse(is.na(dataset$Age),
                     ave(dataset$Age,FUN = function(x) mean(x,na.rm=TRUE))
                     ,dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary),
                     ave(dataset$Salary,FUN = function(x) mean(x,na.rm=TRUE))
                     ,dataset$Salary)


# 2) codificar variables categoricas

dataset$Country = factor(dataset$Country,
                         levels = c("France","Spain","Germany"),
                         labels = c(1,2,3))

dataset$Retired = factor(dataset$Retired,
                         levels = c("No","Yes"),
                         labels = c(0,1))
# Dividir en entrenamiento y test
#install.packages("caTools") instalar packetes en R para cargar packete darle al check 
#library(caTools) cargar la libreria
set.seed(123) # esta semilla va impactar al split
split = sample.split(dataset$Retired,SplitRatio = 0.8)
print(split) # pone en true las que va a usar para training y false las de true
training_set= subset(dataset,split == TRUE)
testing_set= subset(dataset,split == FALSE)
# escalado de valores

# la funcion scale requiere que sean numeros todos los datos por saca la media
# actualmente son Factores no numeros
training_set[,2:3] = scale(training_set[,2:3])# en R 2,3 es incluido
testing_set[,2:3] = scale(testing_set[,2:3])


