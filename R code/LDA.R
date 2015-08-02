
file = read.csv("C:/Users/mguttman/Documents/Technion/Semester_8/Data Mining/project/Dataset_C_transposed_reduced.csv", header = TRUE)

#install.packages("MASS")
library(MASS)
library(plyr)

### 10 -fold CV ###
data <- file

k = 10 #Folds

# sample from 1 to k, nrow times (the number of observations in the data)
data$id <- sample(1:k, nrow(data), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()

sink("C:/Users/mguttman/Documents/Technion/Semester 8/Data Mining/project/lda_result_10fold", append=TRUE, split=FALSE)
for (i in 1:k) {
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(data, id %in% list[-i])
  testset <- subset(data, id %in% c(i))
  # run model
  mymodel <- lda(formula = as.formula("Died~."),data = trainingset)
  plot(mymodel)
  p_train = predict(mymodel,trainingset)
  print(table(p_train$class,trainingset$Died))
  p_test = predict(mymodel,testset)
  print(table(p_test$class,testset$Died))
}
sink()

### 5 -fold CV ###

data <- file

k = 5 #Folds

# sample from 1 to k, nrow times (the number of observations in the data)
data$id <- sample(1:k, nrow(data), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()

sink("C:/Users/mguttman/Documents/Technion/Semester 8/Data Mining/project/lda_result_5fold", append=TRUE, split=FALSE)
for (i in 1:k) {
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(data, id %in% list[-i])
  testset <- subset(data, id %in% c(i))
  # run model
  mymodel <- lda(formula = as.formula("Died~."),data = trainingset)
  plot(mymodel)
  p_train = predict(mymodel,trainingset)
  print(table(p_train$class,trainingset$Died))
  p_test = predict(mymodel,testset)
  print(table(p_test$class,testset$Died))
}
sink()

### LOOCV ###

data <- file

k = 60 #Folds

# sample from 1 to k, nrow times (the number of observations in the data)
data$id <- sample(1:k, nrow(data), replace = TRUE)
list <- 1:k

# prediction and testset data frames that we add to with each iteration over
# the folds

prediction <- data.frame()
testsetCopy <- data.frame()

sink("C:/Users/mguttman/Documents/Technion/Semester 8/Data Mining/project/lda_result_LOOCV", append=TRUE, split=FALSE)
for (i in 1:k) {
  # remove rows with id i from dataframe to create training set
  # select rows with id i to create test set
  trainingset <- subset(data, id %in% list[-i])
  testset <- subset(data, id %in% c(i))
  # run model
  mymodel <- lda(formula = as.formula("Died~."),data = trainingset)
  plot(mymodel)
  p_train = predict(mymodel,trainingset)
  print(table(p_train$class,trainingset$Died))
  p_test = predict(mymodel,testset)
  print(table(p_test$class,testset$Died))
}
sink()

# ensamble
index_list = c(5,8,24,28,42,54)
trainingset <- data[-index_list,]
testset <- data[index_list,]
testset$Died
# run model
mymodel <- lda(formula = as.formula("Died~."),data = trainingset)
plot(mymodel)
p_train = predict(mymodel,trainingset)
print(table(p_train$class,trainingset$Died))
p_test = predict(mymodel,testset)
print(table(p_test$class,testset$Died))

