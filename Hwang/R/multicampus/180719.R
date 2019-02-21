A <- c(1,4,3,3,3,3,3,NA,NA)
B <- c(4,4,3,4,4,5,4,4,NA)
C <- c(4,3,4,3,4,4,3,3,3)

(data.frame(A,B,C))
(mmm <- matrix(c(A,B,C),ncol=3,byrow = F))


myMean <- function(x){
  x <- na.omit(x)
  mean(x)
}

(matrixColMean <- apply(mmm,2,myMean))
SSA <- sum(na.omit((mmm[,1] - matrixColMean[1])^2)) +
sum(na.omit((mmm[,2] - matrixColMean[2])^2)) +
sum(na.omit((mmm[,3] - matrixColMean[3])^2))
SSA



(meanMatrix <- mean(na.omit(mmm)))
(SSE <-sum((meanMatrix - matrixColMean[1:3])^2))
(SST <- SSE + SSA)


# (matrixRowMean <- apply(mmm,1,myMean))
?apply




(y <- c(681, 728, 917, 898, 620, 643, 655, 742, 514, 525, 469, 727, 525,
       454, 459, 384, 656, 602, 687, 360))
length(y)

(x <- rep(c("A", "B", "C", "D"),c(5,5,5,5)))
?rep  
res <- aov(y~x)
summary(res)
# 0.05 보다 작으면 귀무가설을 기각하고 대립가설을 채택한다.
# 귀무가설 : factor (집단 간 변동)이 없다. 집단 간 차이가 없다.
# 대립가설 : 집단 간 변동이 있다. 집단 간 차이가 있다.

# 분산 분석
# 집단 간 분석을 하기 위해 사용한다.
# 



apply(mtcars,1,class)
apply(mtcars,2,class)

(mtcars$cyl <- factor(mtcars$cyl))
mtcars$gear <- factor(mtcars$gear)
(aovm <- aov(mpg ~ cyl, mtcars))
(aovm <- aov(mpg ~ carb, mtcars))
(aovm <- aov(mpg ~ gear, mtcars))
summary(aovm)
# 0.05보다 작으므로 귀무가설을 기각한다.
# 집단 간 특징이 존재한다.


install.packages("gplots", dependencies = T)
library(gplots)
?plotmeans()
attach(mtcars)
?plotmeans
par(mfrow = c(2,2))
plot(mpg~cyl)
plotmeans(mpg~cyl)
plotmeans(mpg~gear)
plotmeans(mpg~carb)



tukey <- TukeyHSD(aovm)
plot(tukey)


mtcars

mtcars
plot()


library(multcomp)
tuk <- glht(aovm, linfct=mcp(cyl="Tukey"))
tuk <- glht(aovm, linfct=mcp(gear="Tukey"))
plot(cld(tuk, level=.05),col="lightgrey")

# 귀무가설 : 분산이 유의미 하지 않다.
# 대립가설 : 유의미하다.
bartlett.test(mpg~cyl, data = mtcars)
bartlett.test(mpg~gear, data = mtcars)
bartlett.test(mpg~carb, data = mtcars)














