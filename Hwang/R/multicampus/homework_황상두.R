# iris
# summary(iris)
# 
# install.packages("doBy")
# library(doBy)
# 
# summaryBy(~ Sepal.Length , iris)
# summaryBy(Sepal.Width + Sepal.Length ∼ Species, iris)
# iris
# ?summaryBy



#1-1
cars
#1-2
colnames(cars)
#1-3
str(cars)
class(cars)
class(cars$speed)
class(cars$dist)


#2-1
x <- c(5,7,8,9)
y <- c(4,8,6,10)
z <- c('A','B','C','D')
#2-2
x[2] < y[3]
#2-3
x %/% y
#2-4
xMatrix <- matrix(x,nrow=2)
#2-5
dataFrame <- data.frame(x=x,y=y,z=z)



#3
readDataFrame <- read.csv("~/Downloads/salary.csv") # header = T
colnames(readDataFrame)
class(readDataFrame)
str(readDataFrame)
?read.csv



#4
install.packages("reshape")
library(reshape)
library(ggplot2)
class(midwest)


#4-1
midwestDataFrame<- as.data.frame(midwest)
colnames(midwestDataFrame)


#4-2
# midwestDataFrame$poptotal <- colnames(total)
midwestDataFrame <- rename(midwestDataFrame,
                           c(poptotal = "total",
                             popasian = "asian"))
midwestDataFrame
midwest
attach(midwestDataFrame)



#4-3
# 정규분포표 2.21에 해당하는 값 나옵니다.
sum(midwestDataFrame$asian) / sum(midwestDataFrame$total) 





#4-4
# class(midwestDataFrame)
# attach(midwestDataFrame)
# midwestDataFrame$PID
# PID
meanSet <- midwestDataFrame$asian / midwestDataFrame$total
result[meanSet > (mean(midwestDataFrame$asian / midwestDataFrame$total))] <- "large"
result[meanSet <= (mean(midwestDataFrame$asian / midwestDataFrame$total))] <- "small"
result
# ifelse(result == T, "large", "small")

        

#4-5
length( meanSet[meanSet > (mean(midwestDataFrame$asian / midwestDataFrame$total))])





