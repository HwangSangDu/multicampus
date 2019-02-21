

## SVM 실습
library(caret)
library(mlbench)
library(e1071)

?svm
data(Sonar)
data(Vowel)
?createDataPartition()
View(Sonar)
View(Vowel)
set.seed(100)

## 
(ind1 <- createDataPartition(Sonar$Class, p= 0.7))
(ind2 <- createDataPartition(Vowel$Class, p= 0.7))


## Create Train Test Data
(train1 <- Sonar[ind1$Resample1,])
(test1 <- Sonar[-ind1$Resample1,])
(train2 <- Vowel[ind2$Resample1,-1])
(test2 <- Vowel[-ind2$Resample1,-1])

View(train1)
(m1 <- svm(Class~., data=train1, cost=100))
summary(m1)
m1$degree
m1$nSV
m1$tot.nSV
m1$labels
m1$SV
(pred1 <- predict(m1, test1))
confusionMatrix(pred1, test1$Class)$overall[1]




m2 <- svm(Class~., train2)
(pred2 <- predict(m2, test2))
accuracy2 <- confusionMatrix(pred2, test2$Class)$overall[1]


(data1 <- read.table("./data1.txt", header = F, sep = ',',encoding = 'UTF-8' ))
names(data1)=c("x","y")
plot(data1)

# lm
m3 <- lm(y~x, data1)
plot(m3)
par(mfrow=c(2,2))
plot(data1)
lines(data1$x, m3$fitted.values, col ='green', lty=3)

# svm
m4 <- svm(y~x, data1, kernel = "linear")
m4 <- svm(y~x, data1, kernel = "polynomial")
m4 <- svm(y~x, data1, kernel = "radial basis")
par(mfrow=c(2,2))
plot(data1)
lines(data1$x, m4$fitted, col ='red', lty=3)
m4$kernel # 2
m2$kernel # 2
?svm
?kernel
# pred <- predict(m1, )




m5 <- svm(y~x,data1,kernel = "radial basis")
plot(data1)
lines(data1$x, m4$fitted, col ='blue', lty=3)
tune.svm()


?tune.svm
?tune
tuneResult <- tune(svm, y~x, data=data1,
                   ranges = list(epsilon = seq(0,1,0.1),
                                 cost = 2^(2:9)))
lines(data1$x, tuneResult$best.model$fitted, 
      col='orange')














minist_tr <- read.csv("mnist_train.csv", header = T)
minist_te <- read.csv("mnist_test.csv", header = T)


library(caret)
names(minist_tr) <- c('Y', paste('V', 1:(28*28),sep=''))
names(minist_te) <- c('Y', paste('V', 1:(28*28),sep=''))
str(minist_tr)
minist_tr$Y <- as.factor(minist_tr$Y)
minist_te$Y <- as.factor(minist_te$Y)

train3 <- createDataPartition(minist_tr$Y, p=0.1)
test3 <- createDataPartition(minist_te$Y, p=0.1)
train4 <- minist_tr[train3$Resample1,]
test4 <- minist_te[test3$Resample1,]

(mat1 <- as.matrix(train4[1,-1], nrow=28))
max(mat1)
# 255 - 0
View(mat1)
mat2 <- mat1/255
mat3 <- matrix(mat2, nrow=28)
str(mat3)
View(mat3)
image(mat3)

train4$Y[1]
str(train4)
dim(train4)
dim(minist_tr)

m5 <- svm(Y~., train4)
pred5 <- predict(m5, test4)
c3 <- confusionMatrix(pred5, test4$Y)
?sapply

train5 <- train4
train5[,-1] <- sapply(train5[,-1], function(x){
  return(x/255)
})
View(train5)




source("https://bioconductor.org/biocLite.R")
biocLite("EBImage")
library("EBImage")


install.packages("readbitmap")
library(jpeg)
img1 <- readJPEG("./img01.jpg")
img001 <- resize(img1, 100, 100)
class(img1) # 칼러 이미지는 3차원
dim(img1) # 3차원 확인 가능하다.
dim(img001)
''












## ?exmaple
data(iris)
attach(iris)

## classification mode
# default with factor response:
(model <- svm(Species ~ ., data = iris))

# alternatively the traditional interface:
x <- subset(iris, select = -Species)
y <- Species
model <- svm(x, y) 

print(model)
summary(model)

# test with train data
pred <- predict(model, x)
# (same as:)
pred <- fitted(model)

# Check accuracy:
table(pred, y)

# compute decision values and probabilities:
pred <- predict(model, x, decision.values = TRUE)
attr(pred, "decision.values")[1:4,]

# visualize (classes by color, SV by crosses):
plot(cmdscale(dist(iris[,-5])),
     col = as.integer(iris[,5]),
     pch = c("o","+")[1:150 %in% model$index + 1])
