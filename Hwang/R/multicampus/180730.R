?sample()
x <- 1:4
# a random permutation
sample(x)
# bootstrap resampling -- only if length(x) > 1 !
data <- sample(x, 100000 ,replace = TRUE, prob = c(3,5,8,1)) 
# 비복원/복원

hist(data)

iris
(nl = nrow(iris))
# 70 프로
set.seed(100)
(ind1 <- sample(1:nl, nl*0.7, replace = F))
(train1=iris[ind1,])
(test1=iris[-ind1,])


table(train1$Species)
table(test1$Species)

# 특정 집단의 수가 매우 적을 수 있다.
# 그룹별로 골고루 들어갈 수 있도록 해주어야 한다.


install.packages("caret")
library(caret)
help(package = 'caret')
??caret

# 층화 추출법

# createDataPartition(y, p=0.7)
?createDataPartition
caret::createDataPartition
iris$Sepal.Length
(ind2 <- createDataPartition(iris$Sepal.Length,p = 0.7))
str(ind2)
(train2 <- iris[ind2$Resample1,])
table(iris$Species)
table(train2$Species)
(ind3 <- createDataPartition(iris$Species, p = 0.7))
(train3 <- iris[ind3$Resample1,])
(test3 <- iris[-ind3$Resample1,])
table(train3$Species)





# 계통 추출법
set1 <- 15
(bet1 <- nl/set1)
nl

# nl까지 bet1 단위로 증가
seq(sample(1:bet1, 1), nl, bet1)




(y1 <- lm(Sepal.Length~., train1[,1:4]))
summary(y1)
SL <- 2.13 + 0.58*SW + 0.65*PL - 0.43*PW
test1
(pred1 <- predict(y1, newdata = test1))

??MSE 
??SSE


(sse <- sum(test1$Sepal.Length - pred1)^2)
(mse <- mean(test1$Sepal.Length - pred1)^2)
(rmse <- sqrt(mse))

?glm
?family

# iteration(반복 횟수 추가)
(y2 <- glm(Sepal.Length ~. , data=iris[,1:4], family = "gaussian"))
summary(y2)




# install.packages("mlbench")
library(mlbench)
library(caret)
(data(Sonar))



View(Sonar) # 단위가 통일되어 있다!!

set.seed(200)

?createDataPartition
# 데이터 분할 70프로의 훈련데이터 30프로의 테스트 데이터를 사용하겠다.
(ind4 <- createDataPartition(Sonar$Class, p =0.7))
(train4 <- Sonar[ind4$Resample1,])
(test4 <- Sonar[-ind4$Resample1,])



?family
## binomial(이항 분포) 기법을 사용하여 선형식을 찾는다.
y4 <- glm(Class~., data = train4, family = "binomial")
(y4$coefficients)

## 선형식이 주어지고 테스트 독립변수로 예측 하겠다. 
(pred4 <- predict(y4, newdata = test4, type='response'))
levels(test4$Class)

# ifelse(pred4 >= 0.5, "R", "M")

## 예측값은 확률(0~1)이며 0.5이상이면 R로 하겠다.
(pred4_1 <- as.factor(ifelse(pred4 >= 0.5, "R", "M")))

## 에측이 얼마나 잘 되었는지 확인 가능하다.
table(test4$Class, pred4_1)

## 성능 평가 함수
?confusionMatrix()
(c1 <- confusionMatrix(pred4_1, test4$Class))

c1$overall
c1$overall[1]

# Accuracy: Overall, how often is the classifier correct?
#   (TP+TN)/total = (100+50)/165 = 0.91
?confusionMatrix()


# KNN(최근접 이웃 유클리드, 맨하탄 거리) , random Forest , DT(결정트리) , SVM, 신경망(딥러닝)

?knn
??knn
library(class)
length(train4)
length(test4)

train4$Class

# knn (최근접 이웃)
# 1.훈련데이터 셋
# 2. 테스트 데이터 셋
# 3. Class(factor)
pred5 <- knn(train4[,1:60], test4[,1:60], train4[,61])
table(pred5, test4$Class)

(c2 <- confusionMatrix(pred5, test4$Class))



data(Vowel)
View(Vowel)

# install.packages()
library(e1071)
?naiveBayes()
(na1 <- naiveBayes(Class~., data=train4))
(pred6 <- predict(na1, test4))


library(caret)
(c2 <- confusionMatrix(pred6, test4$Class))

Vowel
(ind5 <- createDataPartition(Vowel$Class, p=0.7))
(train5 <- Vowel[ind5$Resample1,-1])
(test5 <- Vowel[-ind5$Resample1,-1])


## 경사 하강법
library(nnet)
?multinom()
# 엔트로피 (복잡도)
# decay = 학습율(m1 <- multinom(Class~. , data=train5, maxit=100))

(pred01 <- predict(m1, test5))
(res <- confusionMatrix(pred01, test5$Class))
(accuracy <- res$overall[1])



## knn
library(class)
?knn
(pred02 <- knn(train5[,-10], test5[, -10], train5$Class))
confusionMatrix(pred02 , test5$Class)$overall[1]

## 나이브 베이지안 
# 실수형에 약하다.
# y lable의 개수가 늘면 성능이 떨어진다.
(m3 <- naiveBayes(Class~., train5))
(pred03 <- predict(m3, test5))
confusionMatrix(pred03, test5$Class)$overall[1]




## 의사 결정 트리
# 독립변수가 많으면 제대로 동작하지 않는다.
install.packages("party")
library(party)
(m4 <- ctree(Species~., data=train3))
plot(m4)
?ctree
?ctree_control()

pred04 <- predict(m4, test3)
result <- confusionMatrix(pred04, test3$Species)
result$overall[1]
# 레이블 수가 많아지면 정확도가 낮아진다.
# 변수의 개수를 늘리지 않도록 한다.

train4
m5 <- ctree(Class~., train4)
pred05<- predict(m5, test4)
confusionMatrix(pred05, test4$Class)$overall[1]
# plot(ctree4)



m6 <- ctree(Class ~ ., train5)
plot(m6)
pred06 <- predict(m6,test5)
confusionMatrix(pred06, test5$Class)$overall[1]

## Random Forest
install.packages("randomForest")
library(randomForest)


m7<- randomForest(Class~., train4)
plot(m7)
(pred07 <- predict(m7, test4))
confusionMatrix(pred07, test4$Class)$overall[1]


m8 <- randomForest(Class~., train5)
pred08 <- predict(m8, test5)
confusionMatrix(pred08, test5$Class)$overall[1]















