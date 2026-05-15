##############################################

# Script U6 Actividad - Entrenamiento de un predictor sustitutivo de MammaPrint: Parte II
# "Tue Mar 17 18:43:07 2026"
# Alba Nieto Mayo

###############################################

#install.packages(c("caret", "MLeval","pROC"))
#install.packages("Rtools")


###################################################
# 1) Seleccionamos directorio y cargamos entorno
###################################################
  
setwd("C:/Users/Usuario")

load("./myEnvironment.RData")

library(caret)

#########################################################
# 2) PARTICION DE DATOS Y OPTIMIZACION DE HIPERPARAMETROS
#########################################################

### crear la particion de dataset

# comprobamos equilibrio entre las clases

table (riesgo)

set.seed(123)
trainIdx <- createDataPartition(riesgo, p=0.7, list = FALSE, times = 1)

dataTrain <- filteredVar[trainIdx,]
dataTest <- filteredVar[-trainIdx,]

table(riesgo)
table(riesgo[trainIdx])
table(riesgo[-trainIdx])


### preprocesado de los datos

preProcessPar <- preProcess(dataTrain, method = c("center","scale"))
trainProcessed <- predict(preProcessPar, dataTrain)  # solo predictores
ValProcessed <- predict(preProcessPar, dataTest)     # solo predictores

# definir las clases

clase <- riesgo [trainIdx]    #vector con etiquetas de entrenamiento
claseTest <- riesgo[-trainIdx]    # datos para evaluar el modelo mas tarde

# comprobar que son data frame

trainProcessed<- as.data.frame(trainProcessed)
ValProcessed <- as.data.frame(ValProcessed)


### definir hiperparametros de control de entrenamiento

set.seed(123)
fitCtrl <- trainControl (method = "cv",
                         number = 10,
                         #repeats = 3,
                         verboseIter = TRUE,
                         selectionFunction = "best",
                         search = "grid",
                         savePredictions = TRUE,
                         classProbs = TRUE)



####################################################
# 3) ENTRENAMIENTO DEL MODELO RF
####################################################

#definir hiperparametros del modelo
tuneGridRF <- expand.grid(.mtry = c(5, 50, 75))

trainRF <- cbind(trainProcessed, clase)      # datos para entrenamiento

RF <- train(clase ~ ., data = trainRF,
            method = "rf", 
            trControl = fitCtrl, 
            ntree = 50, 
            verbose = FALSE,
            tuneGrid = tuneGridRF,
            classProb = TRUE)


### CALCULAR PREDICCIONES

# Predicciones de clase
predTestRF <- predict(RF, newdata=ValProcessed, type="raw")

# Predicciones de probabilidad
probRF <- predict(RF, ValProcessed, type = "prob")



### CREAR TABLA MUESTRAS-PREDICCIONES

tabla_pred <- data.frame(
  Muestra = rownames(ValProcessed),
  Prediccion = predTestRF,
  Prob_AltoRiesgo = probRF[, "AltoRiesgo"])

write.csv(tabla_pred, "predicciones_RF.csv")



### CREAR MATRIZ DE CONFUSION

library(caret)
CM <- confusionMatrix(data = predTestRF,
                reference = claseTest,
                positive = "AltoRiesgo",
                mode = "everything")
write.csv(CM$table, "Matriz_confusion_RF.csv")
CM$table


### CURVAS ROC-AUC

#BiocManager::install("MEval")
library(MLeval)
evalRF <- evalm(RF)
evalRF$roc    # ploteamos
evalRF$stdres #hallamos metricas


library(pROC)
rocRF <- roc(
  response = claseTest,
  predictor = probRF[, "AltoRiesgo"],
  levels = c("BajoRiesgo", "AltoRiesgo"),
  direction = "<"
)

plot(rocRF, col = "#1c61b6", lwd = 3, main = "Curva ROC - Random Forest")
auc(rocRF)




###############################################
# 4) ENTRENAMIENTO DEL MODELO GLMNET (LASSO)
###############################################

### definir hiperparametros de tuneado del modelo
tuneGrid_GLM <- expand.grid(alpha = 1,lambda = seq(0.0001, 1, length = 20))


trainGLM <- cbind(trainProcessed,clase) # datos para el entrenamiento


### ENTRENAMIENTO DEL MODELO 

GLM <- train(clase ~ ., data = trainGLM,
             method = "glmnet",
             trControl = fitCtrl,
             tuneGrid = tuneGrid_GLM)

### CALCULAR PREDICCIONES

# Predicciones de clase
predTest_GLM <- predict(GLM, newdata=ValProcessed, type="raw")

# Predicciones de probabilidad
probGLM <- predict(GLM, ValProcessed, type = "prob")

### CREAR TABLA MUESTRAS-PREDICCIONES

tabla_pred_glm <- data.frame(
  Muestra = rownames(ValProcessed),
  Prediccion = predTest_GLM,
  Prob_AltoRiesgo = probGLM[, "AltoRiesgo"])

write.csv(tabla_pred, "predicciones_GLM.csv")


### CREAR MATRIZ DE CONFUSION
CM_GLM <- confusionMatrix(
  data = predTest_GLM,
  reference = claseTest,
  positive = "AltoRiesgo",
  mode = "everything")

write.csv(CM_GLM$table, "Matriz_confusion_GLM.csv")


### CURVAS ROC-AUC

#BiocManager::install("MEval")
library(MLeval)
evalGLM <- evalm(GLM)
evalGLM$roc    # ploteamos
evalGLM$stdres #hallamos metricas


library(pROC)
rocGLM <- roc(
  response = claseTest,
  predictor = probGLM[, "AltoRiesgo"],
  levels = c("BajoRiesgo", "AltoRiesgo"),
  direction = "<")

plot(rocGLM, col = "#1c61b6", lwd = 3, main = "Curva ROC - GLM-LASSO")

auc(rocGLM)




###################################################
# 5) ENTRENAMIENTO DEL MODELO SVM
###################################################

#definir hiperparametros del modelo
tuneGrid_SVM <- expand.grid(C = c(0.25, 0.5, 1),sigma = c(0.001, 0.01, 0.1))


### ENTRENAMIENTO DEL MODELO 

SVM <- train(x = trainProcessed,
             y = clase,
             method = "svmRadial",
             trControl = fitCtrl,
             tuneGrid = tuneGrid_SVM)

### CALCULAR PREDICCIONES

# Predicciones de clase
predTest_SVM <- predict(SVM, newdata=ValProcessed, type="raw")

# Predicciones de probabilidad
probSVM <- predict(SVM, ValProcessed, type = "prob")

### CREAR TABLA MUESTRAS-PREDICCIONES

tabla_pred_svm <- data.frame(
  Muestra = rownames(ValProcessed),
  Prediccion = predTest_SVM,
  Prob_AltoRiesgo = probSVM[, "AltoRiesgo"])

write.csv(tabla_pred, "predicciones_svm.csv")

### CREAR MATRIZ DE CONFUSION
CM_SVM <- confusionMatrix(
  data = predTest_SVM,
  reference = claseTest,
  positive = "AltoRiesgo",
  mode = "everything")

write.csv(CM_SVM$table, "Matriz_confusion_SVM.csv")
CM_SVM$table


### CURVAS ROC-AUC

# CURVA ROC INTERNA
evalSVM <- evalm(SVM)

# CURVA ROC EXTERNA
rocSVM <- roc(
  response = claseTest,
  predictor = probSVM[, "AltoRiesgo"],
  levels = c("BajoRiesgo", "AltoRiesgo"),
  direction = "<")

plot(rocSVM, col = "#1c61b6", lwd = 3, main = "Curva ROC - SVM Radial")
auc(rocSVM)



###############################################
# 6) ENTRENAMIENTO DEL MODELO GBM
###############################################


# Definir hiperparametros del modelo GBM

tuneGrid_GBM <- expand.grid(
  n.trees = c(50, 100, 150),
  interaction.depth = c(1, 3, 5),
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = c(5, 10))


trainGBM <- cbind(trainProcessed, clase)  # datos para entrenamiento


# Entrenamiento del modelo
set.seed(123)
GBM <- train(clase ~ .,
             data = trainGBM,
             method = "gbm",
             trControl = fitCtrl,
             tuneGrid = tuneGrid_GBM,
             verbose = FALSE)

### CALCULAR PREDICCIONES

# Predicciones de clase
predTest_GBM <- predict(GBM, newdata = ValProcessed, type = "raw")

# Predicciones de probabilidad
probGBM <- predict(GBM, ValProcessed, type = "prob")

# Tabla muestras-predicciones
tabla_pred_gbm <- data.frame(Muestra = rownames(ValProcessed),
                             Prediccion = predTest_GBM,
                             Prob_AltoRiesgo = probGBM[, "AltoRiesgo"])
  
  
write.csv(tabla_pred_gbm, "predicciones_GBM.csv")

# Matriz de confusión
CM_GBM <- confusionMatrix(
  data = predTest_GBM,
  reference = claseTest,
  positive = "AltoRiesgo",
  mode = "everything")

write.csv(CM_GBM$table, "Matriz_confusion_GBM.csv")
CM_GBM$table

# CURVA ROC INTERNA
evalGBM <- evalm(GBM)

# CURVA ROC EXTERNA
library(pROC)
rocGBM <- roc(
  response = claseTest,
  predictor = probGBM[, "AltoRiesgo"],
  levels = c("BajoRiesgo", "AltoRiesgo"),
  direction = "<"
)

plot(rocGBM, col = "#1c61b6", lwd = 3, main = "Curva ROC - GBM")
auc(rocGBM)


####################################################
# 7) COMPARACION DE MODELOS
####################################################

#SELECCIÓN DEL UMBRAL ÓPTIMO MEDIANTE EL ÍNDICE DE YOUDEN

coords(rocRF, "best", ret = c("threshold", "sensitivity", "specificity"))
coords(rocGLM, "best", ret = c("threshold", "sensitivity", "specificity"))
coords(rocSVM, "best", ret = c("threshold", "sensitivity", "specificity"))
coords(rocGBM, "best", ret = c("threshold", "sensitivity", "specificity"))


### output

sessionInfo()