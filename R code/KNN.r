data<-read.csv("C:\\Users\\mguttman\\Documents\\Technion\\Semester_8\\Data Mining\\project\\Dataset_C_transposed_reduced.csv",header=TRUE)
set.seed(24)
gp <- runif(nrow(data))
data <- data[order(gp),]
data_train <- data[1: 53, ]
data_test <- data[54:60, ]
data_train_target <- data[1: 53, 1]
data_test_target <- data[54:60, 1]
#install.packages("class")
require(class)
m1<- knn(train = data_train, test = data_train, cl = data_train_target, k=5)
table(data_train_target, m1)
m1<- knn(train = data_train, test = data_test, cl = data_train_target, k=5)
table(data_test_target, m1)


m1<- knn(train = data_train, test = data_train, cl = data_train_target, k=7)
table(data_train_target, m1)
m1<- knn(train = data_train, test = data_test, cl = data_train_target, k=7)
table(data_test_target, m1)



data_train <- data[1: 48, ]
data_test <- data[49:60, ]
data_train_target <- data[1:48, 1]
data_test_target <- data[49:60, 1]
require(class)
m1<- knn(train = data_train, test = data_train, cl = data_train_target, k=5)
table(data_train_target, m1)
m1<- knn(train = data_train, test = data_test, cl = data_train_target, k=5)
table(data_test_target, m1)

m1<- knn(train = data_train, test = data_train, cl = data_train_target, k=7)
table(data_train_target, m1)
m1<- knn(train = data_train, test = data_test, cl = data_train_target, k=7)
table(data_test_target, m1)


# ensamble
index_list = c(5,8,24,28,42,54)
#sink("C:/Users/mguttman/Documents/Technion/Semester_8/Data Mining/project/KNN.txt", append=TRUE, split=FALSE)
for (i in index_list) {
  trainingset <- data[-i,]
  testset <- data[i,]
  data_train_target <- data[-i, 1]
  data_test_target <- data[i, 1]
  testset$Died
  # run model
  m1<- knn(train = trainingset, test = testset, cl = data_train_target, k=7)
  print( m1)
}
#sink()

