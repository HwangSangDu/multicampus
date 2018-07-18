install.packages("Hmisc")
library(Hmisc)
describe(iris)
plot(iris)
plot(iris$Species ~ iris$Sepal.Width , data = iris)


data("cars")
help("cars")

# 출력을 바로 하는 방법 중 하나는 괄호를 이용하는 것이다

# names()[]
# plyr rename 함수
x <- rep(1:100)
x
ifelse (x %% 2 == 0, "even" , "odd")

switch (x[1],
  "1"=print("1111"),
  "2"=print("22222")
)
na.rm = T 


circle <- function(radius){
  radius*radius*pi
}
circle(3)


rectangle <- function(x,y,h){
  return (((x+y)/2) * h)
}
rectangle(3,4,3)


memo <- rep(0,500)
memo
fibo <- function(n){
  if(n <= 1){
    return (1)
  }
  else if(memo[n] != 0){
    return (memo[n])
  }
  else{
    return (memo[n] <- (fibo(n-1) + fibo(n-2)))
  }
}

fibo(333)
memo[332]

# 
# 
# int memo[50];
# int fibonacci(int n) {
#   if (n <= 1) {
#     return n;
#   } else if (memo[n] != 0) {
#     return memo[n];
#   } else {
#     return memo[n] = fibonacci(n-1) + fibonacci(n-2);
#   }
# }
# 





g <- function(x1)
{
  return (function(x2){return (x1+x2)})
}

ff <- g(3)
ff(99)

ls()
rm()





library(MASS)
options(digits=3)
set.seed(1234)
mean <- c(230.7,146.7,3.6)
sigma <- matrix(
  c(15360.8, 6721.2, -60, # 대칭이 아닌 경우
    6721.2, 4700.9, -16.5,
    -47.1, -16.5, 0.3), nrow=3, ncol=3)

?grep 
?sub

?apply
?lapply


?mvrnorm
(mydata <- mvrnorm(10,mean,sigma))
(mydata <- as.data.frame(mydata))
names(mydata) <- c("y","x1","x2")
dim(mydata)
mydata



?lapply

matrix(unlist(lapply(iris[,1:4],mean)), 
       ncol=4, byrow=T)
d <- as.data.frame(matrix(unlist(lapply(iris[,1:4],mean)), 
                   ncol=4, byrow=T))
d

lapply(iris[,1:4],mean)
do.call(cbind, lapply(iris[,1:4],mean))

?do.call
# 왜 느릴까요

x <- sapply(iris[,1:4],mean)
x
trans <- t(x)
trans
as.data.frame(x)
as.data.frame(t(x))


# data type check
sapply(x, class)


# 범주형 데이터를 그룹화시킬 때 편리하다.
# tapply(1:100, paste(printf("x%3d"), , sep=""),sum)
?tapply






rnorm(10000,0,10000)

?scale()


# 매개변수
# 반환 값

# 매개변수 : 배열, 행렬, 콜백함수
# 반환값  : 벡터, 배열
# 기능 : 콜백함수 적용
? apply 


# 반환값 : 리스트 형태 unlist로 벡터 형태로 변형 가능

?lapply(list, function)
?sapply(list, function)
?sapply

# index 기준으로 그룹화 시킨다.
?tapply
?tapply(vector, index, function)

# 반환값 : 리스트
mapply(rep, 1:4, 4:1)






summary(iris)





options(digits=2)
Students <- c("John Davis", "Angela Williams", "Bullwinkle Moose",
              "David Jones", "Janice Markhammer", "Cheryl Cushing", "Reuven Ytzrhak", "Greg Knox", "Joel England",
              "Mary Rayburn")
Math <- c(502, 600, 412, 358, 495, 512, 410, 625, 573, 522)
Science <- c(95, 99, 80, 82, 75, 85, 80, 95, 89, 86) 
English <- c(25, 22, 18, 15, 20, 28, 15, 30, 27, 18) 

(roster <- data.frame(Students, Math, Science, English,stringsAsFactors = FALSE))
# roster : 점수를 데이터 프레임으로 만듬

(z <- scale(roster[,2:4]))
# z : scale 함수로 표준화 시킴

(score <- apply(z, 1,mean)) # 1 : row 2 : col
# score : 정규화된 z를 행기준으로 평균을 낸다.

(roster <- cbind(roster,score))
# 오른쪽 열에 score을 붙인다.

# score를 사분위수 함수로 표현 
(y <- quantile(score, c(0.8, 0.6, 0.4, 0.2)))

# A : 0.8 ~ 1 
# B : 0.6 ~ 0.8
# C : 0.4 ~ 0.6
# D : 0.2 ~ 0.4
# F : 0 ~ 0.2

roster$grade[score >= y[1]] <- "A"
roster$grade[score >= y[2]] <- "B"
roster$grade[score >= y[3]] <- "C"
roster$grade[score >= y[4]] <- "D"

sapply( , "[", 2)

grep()
?sapply

mtcars

install.packages("doBy")

search()

library(KoNLP)

iris
aggregate(Sepal.Width~Species,iris,FUN=mean)#,FUN=mean, by = list(iris$Petal.Width, iris$Sepal.Width),na.rm=T) 
?aggregate


myiris <- iris
str(myiris)

install.packages("reshape2")
library(reshape)
library(reshape2)
iris
melt(id = 1:4, iris)
melt(id = c(1,2,3), iris[-5])
dcast
?mad

rm(list=ls())
iris

mtcars$mpg
plot(mtcars$mpg)

cars
plot(cars)















employee <- read.csv("salary.csv", header = T)
head(employee)
hist(employee$salary)
boxplot(salary ~ year, data = employee)
hist(employee$salary[employee$negotiated == T])
hist(employee$salary[employee$negotiated == F])

hist(employee$salary[employee$gender == "M"])
hist(employee$salary[employee$gender == "F"])


mystats <- function(x, na.omit=F){
  if(na.omit)
  {
    x <- x[!is.na(x)]
  }
  
  m <- mean(x)
  n <- length(x)
  s <- sd(x)
  
  skew <- sum((x-m)^3/s^3)/n
  kurt <- sum((x-m)^4/s^4)/n-3
  return (c(n=n, mean=m, skewness=skew, kurtosis=kurt))
}

sapply(employee[myvars], mystats)

library(Hmisc)
install.packages("pastecs")
library(pastecs)
myvars <- c("incentive", "salary")
employee[myvars]
employee
describe(employee[myvars])
stat.desc(employee[myvars], desc = F)
stat.desc(employee[myvars], desc = T)


prop.table() # 확률
margin.table # sum


library(vcd)
Arthritis
xtabs(Improved + Treatment, data = Arthritis)




summary(employee)
class(employee)

head(employee)
sapply(employee, class)
Hmisc::describe(employee[myvars])
psych::describe(employee[myvars])

stat.desc(employee)

length(employee)

employee$salary[employee$gender == "M"]
employee$salary[employee$gender == "F"]
xtabs(  ~ employee$gender + employee$salary  + employee$incentive , data=employee)
xtabs(  ~ employee$year + employee$salary  + employee$incentive , data=employee)
mytable <- xtabs(~year + gender, employee)
margin.table(mytable, 1)
margin.table(mytable, 2)



# prop.table로 정돈시킨다.
# ?mosiacplot  범주형




credit_df <- read.csv("credit.csv", header=T)
head(credit_df)
(freeq_Age <- xtabs(~Age, data=credit_df))
margin.table(freeq_Age)
prop.table(freeq_Age)

cross_Age_Credit <- xtabs()



# SE standard ERROR (표본 오차)
# summary(c(260,265,250,270,272,258,262,268,270,252))
data<- c(260,265,250,270,272,258,262,268,270,252)
n <- length(data)
sd(data)
summary(data)
mean(data)
?t
?gt
?pt


1/dt(400,df=n-1)
qt(.975, df=n-1)
rt(100,df=n-1)
pt(seq(from = 0, to = 6 , by=0.01), df = n-1)


?pt
myvar <- stat.desc(data)
mylist <- as.list( myvar)




  
  
sqrt(10)


zstat<- (-0.4 * sqrt(10) * 2)

# pt(zstat , df = 9)
qt(0.05 , df = 9)  > zstat
pt(zstat, df = 9)  
  
  

# 1. 사람은 주관적이라서 불편성보다는 편향된다.
# 2. 그러다 보면 분산이 작게 나오게 된다.
# 3. 이를 수정하기 위해서 n / n-1 을 곱하주게 된다.
# 4. 이것이 수학적 약속이 되었다.




# 범주형 데이터에서 카이제곱 분석 시 많이 사용한다.
# 교차분석
# 카이제곱 검정
# 독립성 검정



(creditName <- c("bad","excellent","fair")
(genderName <- c("M","F"))


credit_df



matrix(data=c("26")




?prob.table







library(vcd)
Arthritis







(cross_Income_Credit <- xtabs(~Income+Credit, data = credit_df))
(chi_test_cross <- chisq.test(cross_Income_Credit))
myt <- matrix(c(26,23,10,28,72,70), ncol = 3, nrow = 2, dimnames = list(genderName,
                                                                 creditName))
prop.table(myt, margin = 1)
prop.table(myt, margin = 2)

barplot(myt, beside=T)


?matrix
chi_test_cross$observed

chisq.test(myt)



(data <- c(62, 42, 37, 61, 35, 72, 37, 76, 28, 71, 57, 24))
(myMartrix <- matrix(data , nrow = 3 , ncol = 4, dimnames = list(c('i','ii','iii'),c('A','B','C')))
myMartrix




















