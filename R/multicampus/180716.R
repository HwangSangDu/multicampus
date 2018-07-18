
# 2 X 2 그래프 보기
par(mfrow=c(2,2))
Hmisc::describe(data2)
mtcars
View(mtcars)

(data2 <- read.csv('salary.csv'))

# 히스토그램
hist(data2$salary)
hist(data2$incentive)
summary(data2)
?xtabs
# 산점도
plot(data2$salary,data2$incentive)#, xlab = , xlim = ) 생략
mmm<- xtabs(~data2$gender+year, data = data2)
(xtabs(~gender + year, data = data2))
mmm
barplot(xtabs(~data2$gender+year, data = data2))
?xtabs
# xtabs(gender ~ year , data = data2)


# 막대 그래프
?hist
?barplot
# barplot(data2[, c("year", "gender")])
hist(data2)
library(ggplot2)

# summary()
# Hmisc::describe()
# stat.desc()
# aggregate()



myFunc <- function(x){
  dim(x)
  (str(x))
  (head(x))
  (class(x))
  (apply(x, 2,class))
  (plot(x))
  (summary(x))
  # (boxplot(x))
}





library(corrplot)

corrplot(cor(mtcars[,1:length(mtcars)]),method=c("ellipse"))

corrplot(cor(iris[,1:4]),method=c("ellipse"))


str(Titanic)
(mmm <- as.data.frame(Titanic))
dim(Titanic)
plot(mmm$Class~mmm$Age)
xtabs(~Freq + Sex, mmm)
barplot(t(xtabs(~Freq + Sex, mmm)))



corrplot(cor(Titanic),method=c("ellipse"))

plot(cars)
corrplot(cor(cars), method=c("ellipse"))

corrplot(cor(Arthritis), method=c("ellipse"))

?prop.table
prop.table(cars)
prop.table(mtcars)




# apply(Titanic,2,class)
?corrplot
corrplot(cor(data))
(data <- as.vector(Titanic))
Titanic
barplot(data)


mtcars
addmargins(mtcars,c(1,2))
# (head(cars))
# (str(cars))
# (class(cars))
# (apply(cars, 2,class))
# (plot(cars))
chisq.test(mmm)


plot(mtcars)

myFunc(mtcars)
myFunc(cars)
myFunc(Titanic)
myFunc(iris)


(states <- data.frame(state.region, state.x77))
myFunc(states)
myFunc((midwest))



# Arthritis
library(vcd)
Arthritis
myFunc(Arthritis)
(mmm <- xtabs(~Sex , Arthritis))
(mmm <- xtabs(~Sex + Treatment , Arthritis))
barplot(mmm)





cars
mtcars

View(cars)

str(cars)
str(mtcars)
?cars



# 1. boxplot으로 이상치를 제거한다.
# 2. 서로 독립적이고 정규분포를 따른다고 가정한다.

A <- c(12.6, 15.15, 17.62, 16.81, 15.51, 15.12, 14.39, 15.20, 13.70, 14.75,
       15.13, 15.66, 13.69, 15.74, 14.96, 15.20, 16.45, 13.66, 16.16, 14.74)
B <- c(13.77, 13.63, 12.63, 14.13, 13.50, 13.09, 13.96, 13.41, 14.03, 14.25,
       13.47, 13.43, 13.24, 14.61, 13.82, 14.07, 15.96, 13.69, 14.25, 14.50)
length(A)
length(B)

boxplot(A,B, names = c("Cars.A", "Cars.B"))
boxA <- boxplot(A)
boxB <- boxplot(B)
boxA$out
boxB$out

?boxplot
boxA$out
boxB$out
(A <- A[A != boxA$out])
(B <- B[B != boxB$out])

boxplot(A,B, names = c("Cars.A", "Cars.B"))

(mean(A)-mean(B))
(diff<- (mean(A) - mean(B)))
(mysd <-  sqrt ((sd(A)^2 / length(A)  + sd(B)^2 / length(B))))



(staticZ <- diff / mysd)


(qnorm(1 - (0.05/2),sd=1))

?qnorm()





cars
boxplot(cars)
var.test(cars$speed,cars$dist)









u2 <- c(102, 117, 82, 104,
77, 110, 93 ,115,
75, 103, 126, 79,
81, 118, 93 ,106,
104, 97, 115, 80,
78, 109, 116, 104,
102, 137, 99, 100,
113, 112, 96, 106,
76, 102, 111, 105,
85, 125, 77, 111)


u1 <- c(91, 115, 96, 90,
120, 108, 82, 118,
105, 97, 113, 119,
90, 106, 116, 92,
108, 115, 114, 101,
96, 96, 96, 97,
89, 99, 90 ,85,
91 ,124, 93, 90,
100, 100, 91, 96,
120, 78, 96, 114)


mean(u1)
mean(u2)

sd(u1)
sd(u2)



p <- 0.05
n1 <- length(u1)
n2 <- length(u2)
(deno <- sqrt(100 / n1 + 225 / n2))
(diff <- mean(u1) - mean(u2))
(zStatic <- diff / deno)


zp <- (qnorm(1-(p/2), sd = 1))

(right <- zStatic * zp + diff)
(left <- zStatic * zp - diff)
(right - left)






# 2-1
u1 <- c(39.1, 38.4, 32.5,
42.3, 43.9, 39.6,
40.2, 38.0, 42.3,
34.6, 41.4, 38.7,
41.9, 37.9, 36.9)
u1
u2 <- c(55.3, 50.2, 47.0,
65.2, 44.0, 46.7,
48.8, 36.6, 57.5,
51.1, 46.7, 42.6,
65.6, 56.4, 38.4)
u2
?boxplot
# 2-2
box1 <- boxplot(u1)
box2 <- boxplot(u2)
box <- boxplot(u1,u2, names = c("A", "B"))

# 이상치가 없습니다.
# (u1_corrext <- u1[u1 != box$out[1]])


# 2-3
# 평균,  분산
mean(u1)
mean(u2)
sd(u1)
sd(u2)



# 2-4
?shapiro.test
# 매개변수 : numeric 벡터
# 기능 : 모집단이 정규분포를 따르는지 알 수 있다.

# 유의수준 5프로 기준으로
shapiro.test(u1) # 정규분포 안 따르네요
shapiro.test(u2) # 정규분포 따르네요



# 2-5
# 카이스퀘어 분포를 이용해서
# 두 모집단의 분산이 동일한지 검사합니다.
?var.test
var.test(u1,u2)
?chisq.test
chisq.test(u1)

t.test(u1,u2)
diff <- mean(u1) - mean(u2)
var_mean <- sqrt(sd(u1)^2/length(u1) +sd(u2)^2/length(u2))
zStatic <- diff / var_mean

p <- 0.05
z.alpha <- qnorm(1- (0.05/2), sd = 1)
(upper <- diff +  var_mean*z.alpha)
(lower <- diff - var_mean*z.alpha)


(p_value_R <- pnorm(zStatic, lower.tail = FALSE))
(p_value_L <- pnorm(zStatic, lower.tail = TRUE))
(p_value_two <- pnorm(abs(zStatic), lower.tail = FALSE))


?pnorm


# create data
before <- c(71,72,66,69,69,69,70,67,72,67,71,72,69,69,70)
after  <- c(69,67,68,68,70,67,67,64,65,64,66,70,70,67,66)

# boxplot
box <- boxplot(before, after, names = c("before", "after"))


# 유의 수준 0.05
shapiro.test(before) 
shapiro.test(after) 

mean(before)
mean(after)
sd(before)
sd(after)

diffMean <- mean(after - before)


D <- after - before
boxplot(D)
mean(D)
sd(D)

t.test(before,after, paired =T)
t.test(after, before, paired =T)



data <- c(294, 6, 390, 10)
(mmm<-matrix(data, ncol = 2, byrow = T))


#1 
ttt<- prop.table(mmm)
barplot(t (ttt))

#2
(propability<- prop.table(mmm,margin = c(1)))
# propability[1,2] * (1 - propability[1,2])
# propability[1,1] * (1 - propability[1,1])
lineSize1 <- margin.table(mmm, margin = 2)[1]
lineSize2 <- margin.table(mmm, margin = 2)[2]
(var_mean <- (propability[1,2] * (1 - propability[1,2]) / lineSize1) +
  (propability[1,1] * (1 - propability[1,1]) / lineSize2))


(diff<- propability[1,2] - propability[1,1])
(diff / sqrt(var_mean))
prop_test <- prop.test(mmm)
summary(prop_test)
prop_test$statistic
prop_test$estimate



prop.test(t(mmm))
mmm

?prop.table


library(MASS)
head(UScrime)
apply(UScrime,2,class)
UScrime

cars
# 결과 :  y절편 , x 기울기
(m <- lm(dist ~ speed , cars))

?coef
coef(m)



?fitted 
cars
fitted(m)




(x <- c(1095, 1110, 1086, 1074, 1098, 1105, 1163, 1124, 1088, 1064))
length(x)
(y <- c(49,52,48,49,50,51,50,51,49,48))
length(y)



?lm
(m <- lm(y[c(-7)]~x[c(-7)]))
# m$coefficients
# m$residuals
# m$effects


coef(m)
# 적합화 된 값
fitted(m)[1:4]
# 잔차
residuals(m)[1:4]
# 잔차 제곱합 (SSE)
deviance(m)
# 
?confint
confint(m)

?predict
predict(m, newdata = data.frame(x=1003))
predict(m, newdata = data.frame(x=1200), interval = "prediction")
predict(m, newdata = data.frame(x=1200), interval = "confidence")


summary(m)
plot(m)












par(mfrow = c(2,2))












# 실습 2
x <- c(26, 16, 20, 7, 22, 15, 29, 28, 17, 3, 1, 16, 19, 13, 27, 4, 30, 8, 3, 12)
y <- c(1267, 887, 1022, 511, 1193, 795, 1713, 1477, 991, 455,324, 944, 1232, 808, 1296, 486, 1516, 565, 299, 830)
m2 <- lm(y[-7]~x[-7])



# 회귀 계수
coef(m2)
summary(m2)

# 신뢰구간
confint(m2)
# SSE
deviance(m2)
# 적합화 된 값
fitted(m2)[1:4]
# 잔차
residuals(m2)[1:4]


# 예측
predict(m2, newdata = data.frame(x=1003))






plot(m2)
anova(m2)




