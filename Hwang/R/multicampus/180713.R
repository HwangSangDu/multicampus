
(data <- c(62, 40, 37, 61, 35, 72, 37, 76, 28, 71, 57, 24))
length(data)
(mmm <- matrix(data , nrow = 3 , ncol = 4, byrow = T,dimnames = list(c("i","ii","iii"),c("A","B","C","D"))))
addmargins(mmm)
prop.table(mmm)
prop.table(mmm, margin = 1)
prop.table(mmm, margin = 2)
barplot(mmm, beside=T, col = c("red", "green", "blue"))
chisq.test(mmm)
library(Hmisc)
describe(iris)
describe(mmm)

apply(mmm, c(1),mean)
apply(mmm, c(2),mean)



install.packages("corrplot")
corrplot::corrplot(cor(iris[,1:4]), method= "ellipse")



year <- c(2001:2015)
year <- as.character(year)
class(year)
year <- as.vector(year)
advertise <- c(13,8,10,15,12,15,14,15,17,19,20,21,22,21,25)
expense <- c(94,70,90,100,95,100,85,95,105,105,110,105,104,105,121)
length(advertise)
length(expense)
mmm <- matrix(c(advertise , expense) , nrow = 2, dimnames = list(c("ad" , "ex"),year))
?matrix
mmm


plot(advertise, expense)
plot(mmm[1,], mmm[2,])

corrplot::corrplot(cor(data.frame(t(mmm))), method= "ellipse")
?data.frame()


(data <- state.x77[,1:6])

library(corrplot)
?cov
?corrplot
cov(data)
corrplot(cor(data), method="ellipse")
plot(data[,4],data[,5])
cor(data[,4],data[,5])
cor(data)

cor.test(data[,4],data[,5], method="kendall")
cor.test(data[,4],data[,5])
cor.test(data[,4],data[,5], method="spearman", conf.level = 0.96)

?cor.test
corrplot()































