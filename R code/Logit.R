file = read.csv("C:/Users/mguttman/Documents/Technion/Semester_8/Data Mining/project/Dataset_C_transposed_reduced.csv", header = TRUE)

#trainInd = sample(nrow(file),floor(nrow(file)*0.8))
#train = file[trainInd,]
#test = file[-trainInd,]


splitData <- function(dataSet, seed)
{
 set.seed(seed)
 smp_size <- floor(0.8 * nrow(dataSet))
 train_ind <- sample(seq_len(nrow(dataSet)), size = smp_size)
 train <- dataSet[train_ind, ]
 test <- dataSet[-train_ind, ]
 list(train=train,test=test)
}

splittedData = splitData(file, 218)

train = splittedData$train
test = splittedData$test


glm_model = glm(formula = as.formula("Died~."),data = train, family = binomial("logit"))


#install.packages("ROCR")
#library(ROCR)


## running predict on train set
#glm_result_train = predict(glm_model)
## ROC plot for train set
#pred_train <- prediction(glm_result_train, train$Died)
#perf_train <- performance(pred_train, measure = "tpr", x.measure = "fpr") 
#plot(perf_train, colorize = T)
#
## running predict on test set- no specific tau - probably not needed!
#glm_result_test = predict(glm_model, newdata = test)
## ROC plot for test set- probably not needed!
#pred_test <- prediction(glm_result_test, test$Died)
#perf_test <- performance(pred_test, measure = "tpr", x.measure = "fpr") 
#plot(perf_test, colorize = T)


#Odelya - this function calculate the  
roc.curve=function(tau,print=FALSE){
 Ps=(S>tau)*1
 Specificity=sum((Ps==0)*(cur_data$Died==0))/sum(cur_data$Died==0)
 Sensitivity=sum((Ps==1)*(cur_data$Died==1))/sum(cur_data$Died==1)
 if(print==TRUE){
  print(table(Observed=cur_data$Died,Predicted=Ps))
 }
 vect=c(tau,Specificity,Sensitivity)
 names(vect)=c("tau","Specificity","Sensitivity")
 print(vect)
 return(vect)
}


sink("C:/Users/odelya/Desktop/DM_Project/LR/logistic_with_tau_train", append=TRUE, split=FALSE)

#ROC curve for train data
cur_data=train
S=predict(glm_model,type="response",newdata=cur_data)
for (threshold in seq(0,1,by=0.1)) {
    roc.curve(threshold,print=TRUE)
}
sink()

sink("C:/Users/odelya/Desktop/DM_Project/LR/logistic_with_tau_test", append=TRUE, split=FALSE)
#ROC curve for test data
cur_data=test
S=predict(glm_model,type="response",newdata=cur_data)
for (threshold in seq(0,1,by=0.1)) {
    roc.curve(threshold,print=TRUE)
}
sink()


# ensamble
data <- file
index_list = c(5,8,17,28,42,54)
trainingset <- data[-index_list,]
testset <- data[index_list,]
testset$Died
# run model
glm_model = glm(formula = as.formula("Died~."),data = trainingset, family = binomial("logit"))

sink("C:/Users/mguttman/Documents/Technion/Semester_8/Data Mining/project/logistic_with_train.txt", append=TRUE, split=FALSE)
#ROC curve for train data
cur_data=trainingset
S=predict(glm_model,type="response",newdata=testset)
for (threshold in seq(0,1,by=0.1)) {
  roc.curve(threshold,print=TRUE)
}
sink()

sink("C:/Users/odelya/Desktop/DM_Project/LR/logistic_with_tau_test", append=TRUE, split=FALSE)
#ROC curve for test data
cur_data=test
S=predict(glm_model,type="response",newdata=cur_data)
for (threshold in seq(0,1,by=0.1)) {
  roc.curve(threshold,print=TRUE)
}
sink()