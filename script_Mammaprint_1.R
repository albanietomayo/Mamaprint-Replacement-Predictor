##############################################

# Script U5 Actividad 1 - Entrenamiento de un predictor sustitutivo de MammaPrint: Parte I
# "Thu Mar 12 21:11:38 2026"
# Alba Nieto Mayo

###############################################



##############################################
### 0) SETUP (Paquetes y carpetas)
##############################################

### Instalar paquetes de Bioconductor ##

#BiocManager::install(version = "3.14")
#BiocManager::install("genefu")

#install.packages("devtools")
#devtools::install_github("bhklab/genefu")
#install.packages(c("caret", "MLeval","pROC"))
#install.packages("Rtools")

#if(!require("caret")) install.packages("caret")
#install.packages(c("MLeval","pROC"))

#if (!require("BiocManager")) install.packages("BiocManager")
#BiocManager::install(version = "3.22", force = TRUE)

#if (!require("genefu")) BiocManager::install("genefu")


### Definimos las rutas de directorios

data_folder <- "C:/Users/Usuario" ## Ruta a la carpeta del proyecto o de los datos
OUT_DIR <- file.path(data_folder, "outputs")
dir.create(OUT_DIR, recursive = TRUE)



### Estructura del directorio:


#   ./clinical_info_TCGA-BRCA_filtered.tsv
#   ./data_norm.tsv  
#   ./matriz4ML_MammaPrint.tsv
#   ./Script Actividad 1 U5.R   (este script)
#   ./outputs/                 (out_dir)

getwd()   
setwd("C:/Users/Usuario")   

list.files()


#############################################################
### 1) Analisis y visualizacion de la estructura de los datos
#############################################################

### carga de datos

clinical <- read.delim("./clinical_info_TCGA-BRCA_filtered.tsv", sep = "\t", 
                       check.names = T,
                       header = TRUE, fill = TRUE, stringsAsFactors = F) 
mat<-read.delim("./data_norm.tsv", header = TRUE, 
                sep = "\t", fill = TRUE)


### ver cuantas muestras de cada clase

table(clinical$er_status_by_ihc) 
sum(clinical$er_status_by_ihc == "Positive")


### Seleccionamos filas positivas (ER+) y guardamos la matriz

clinical <- clinical[clinical$er_status_by_ihc == "Positive",]

write.table(clinical, "matriz_clinical.tsv", sep="\t", row.names=FALSE)


###comprobamos si ambas matrices tienen las mismas muestras
all(rownames(clinical) == colnames(mat))


###seleccionamos las muestras en comun
matfiltered <- mat[,rownames(clinical)]

###comprobamos de nuevo si las matrices tienen las mismas muestras
all(rownames(clinical) == colnames(matfiltered))


#######################################################
# 2) Determinacion de MammaPrint
#######################################################

### Anotacion de genes

library(genefu)
library(AnnotationDbi)
library(org.Hs.eg.db)

### Trasposicion de la matriz
mat <- t(matfiltered)

### sacar EntrezID

entrez <- mapIds(org.Hs.eg.db, keys = colnames(mat), column = c("ENTREZID"), keytype = "SYMBOL")

all(names(entrez)==colnames(mat)) #comprobar similitud entre genes

entrez <- entrez[!is.na(entrez)] #eliminamos NAs
mat<-mat[,!is.na(entrez)]

### crear la matriz de anotacion
entrezIds <- cbind("EntrezGene.ID"=entrez, "Symbol"=colnames(mat)) 


### Calculo de riesgo con MP
risk <- gene70(data=mat,annot = entrezIds, do.mapping = T, verbose = T)

###creamos una columna de riesgo y otra de sccore en clinical
clinical$MP_risk <- risk$risk
clinical$MP_score <- risk$score
table(clinical$MP_risk)
write.table(clinical$MP_risk, "clinical$MP_risk.tsv", sep="\t", row.names=FALSE)



##################################################
#3) ENTRENAMIENTO DEL MODELO
##################################################


### carga de archivos para el entrenamiento del modelo
  
setwd("C:/Users/Usuario")   
clinical <- read.delim("./clinical_info_TCGA-BRCA_MammaPrintInfo.tsv", header = T, check.names = T, 
                       fill = T, sep = "\t", stringsAsFactors = F)

matML<- read.delim("./matriz4ML_MammaPrint.tsv", header = T, check.names = T, 
                   fill = T, sep = "\t", dec = ".", stringsAsFactors = F)

table (clinical$MP_risk)
write.table(clinical$MP_risk, "clinical$MP_risk.tsv", sep="\t", row.names=FALSE)



### definimos la etiqueta de clase
riesgo <- factor(clinical$MP_risk, levels = c(0,1), labels = c("BajoRiesgo","AltoRiesgo"))

library(caret)


### visualizacion de las caracteristicas (genes)

featurePlot(x = matML[,1:6], y = clinical$MP_score,
            plot = "scatter", auto.key = list(columns=2))

dev.new() #abrir el plot en nuevo dispositivo
featurePlot(x = matML[,1:6], y = riesgo,
            plot = "pairs", auto.key = list(columns=2))

featurePlot(x=matML[,1:6], y=riesgo, 
            plot = "box")

dev.new() 
featurePlot(x=matML[,1:6], y=riesgo, 
            plot = "density", auto.key = list(columns=2))
dev.off()


### calcular variables dummy y guardar matrices

newMat <- cbind(riesgo,matML)

dumVar <- dummyVars(~ riesgo, data = newMat)
dummy_riesgo <- predict(dumVar, newdata = newMat)
dummy <- cbind(dummy_riesgo, matML)

write.table(dummy, "matriz_dummy", sep="\t", row.names=FALSE)
write.table(newMat, "matriz_newMat", sep="\t", row.names=FALSE)


### comprobacion de matrices
dim(dummy)
dim(newMat)


### filtrar caracteristicas de varianza cero o casi cero

nzv <- nearZeroVar(matML,saveMetrics = T)
filteredMat <- matML[,-nzv$nzv]
write.table(filteredMat, "matriz_varianza_cero.tsv", sep="\t", row.names=FALSE)

### filtrar caracteristicas por alta correlacion

varCor <- cor(filteredMat)
highCorVar <- findCorrelation(varCor, cutoff = 0.75)
filteredVar <- filteredMat[,-highCorVar]
write.table(filteredVar, "matriz_alta_correlacion.tsv", sep="\t", row.names=FALSE)

### comprobamos de nuevo la correlacion 

varCor2 <- cor(filteredVar)
summary(varCor2)


### guardamos entorno

#rm(list = c("")) #eliminar objetos del entorno
save.image(file = "myEnvironment.RData")
saveRDS(filteredVar, file = "FilterVar.rds") #guardar objeto para parte II

#readRDS()
#load()


### output

sessionInfo()


