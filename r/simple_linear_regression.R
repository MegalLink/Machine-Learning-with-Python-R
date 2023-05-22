
dataset = read.csv('./datasets/regression/salary_data.csv')
set.seed(123)
#use field of prediction
split = sample.split(dataset$Salary,SplitRatio = 0.8)
training_set= subset(dataset,split == TRUE)
testing_set= subset(dataset,split == FALSE)

linear_model = lm(formula = Salary ~ YearsExperience,data = training_set)

#predctions testing set columns has to use the same columns as model trained
prediction_y= predict(linear_model,newdata= testing_set)
#install.packages("ggplot2")# instalar packetes en R para cargar packete darle al check 
#library(ggplot2) #cargar la libreria
ggplot() +
  geom_point(aes(x=training_set$YearsExperience,y=training_set$Salary),
             colour="red") +
  geom_line(aes(x=training_set$YearsExperience,
                y=predict(regressor,newdata = training_set)),colour="blue") +
  ggtitle("Sueldo vs Años de experiencia (Conjunto de entrenamiento)") +
  xlab("Años de experiencia") +
  ylab("Sueldo (en $)")

# testing predictions
ggplot() +
  geom_point(aes(x=testing_set$YearsExperience,y=testing_set$Salary),
             colour="red") +
  geom_line(aes(x=training_set$YearsExperience,
                y=predict(regressor,newdata = training_set)),colour="blue") +
  ggtitle("Sueldo vs Años de experiencia (Conjunto de testing)") +
  xlab("Años de experiencia") +
  ylab("Sueldo (en $)")
