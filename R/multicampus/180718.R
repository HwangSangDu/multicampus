install.packages("UsingR")
library(UsingR)


(child <- c(rep(61.7,5) , rep(62.7,6)))
(parent <- c(70.5, 68.5, 65.5, 64.5, 64.0,rep(67.5,3), rep(66.5,3)))
(m <- lm(galton))



# 그래프 2 X 2
par(mfrow=c(2,2))
# 회귀 계수
coef(m)
summary(m)

# 기울기 및 y절편 신뢰구간
confint(m)

# SSE (표준화된 오차 및 잔차) 제곱 합
deviance(m)

# 적합화 된 값 (만들어진 회귀분석 1차함수로 독립변수를 기준으로 종속변수 예측?)
fitted(m)

# 각 요소별 잔차
residuals(m)

?predict()

# 독립션수를 기준으로 예측하기!
# y 절편
# confidence interval : 신뢰구간
predict(m , newdata = data.frame(child=c(0)))
predict(m , interval = "confidence")
predict(m , newdata = data.frame(child=c(rep(0:100))))

# (m <- lm(parent[c(-2,-1,-3)]~child[c(-2,-1,-3)]))
plot(m)


# # 회귀 계수
# coef(m2)
# summary(m2)
# 
# # 신뢰구간
# confint(m2)
# # SSE
# deviance(m2)
# # 적합화 된 값
# fitted(m2)[1:4]
# # 잔차
# residuals(m2)[1:4]
# 
# 
# # 예측
# predict(m2, newdata = data.frame(x=1003))






# 정규성 검사
hist
shapiro.test(residuals(m))
?qqPlot

View(women)
?poly
mm <- lm(women$height ~ poly(women$weight,2))
mmm <- lm(women)
summary(mm)
summary(mmm)

shapiro.test(residuals(mm))
shapiro.test(residuals(mmm))

plot(mm)
plot(mmm)


x <- c(0.04, 0.07, 0.11, 0.13, 0.20, 0.27, 0.39, 0.42, 0.52, 0.56,
       0.61, 0.75, 0.78, 0.86, 0.89, 0.92, 0.94, 0.97, 0.98, 0.99)
y <- c(1.42, 1.41, 1.37, 1.34, 1.26, 1.21, 1.13, 1.08, 1.05, 1.04,
       1.02, 0.96, 0.98, 0.97, 0.98, 0.97, 0.99, 0.98, 0.98, 0.98)

plot(x,y)

m2<- lm(y~x)
m3 <- lm(y~poly(x,2,raw=T))


m2$coefficients
m3$coefficients

curve(m)
abline(m3$coefficients)

summary(m2)
plot(m2)

summary(m3)
plot(m3)




# 









x1 <- c(507,391, 488, 223, 274, 287, 550, 457, 377, 101,
        170, 450, 309, 291, 375, 198, 641, 528, 500, 570)

x2 <-c("F","F","F","F","F","F","F","F","F","F",
       "M","M","M","M","M","M","M","M","M","M")

y <- c(1096, 759, 965, 698, 765, 703, 968, 805, 912, 588,
       281, 527, 439, 318, 412, 370, 1044, 537, 649, 807)

plot(x1, y, pch=x2)

except <- c(-1,-17,-18,-20)
tlm <- lm(y[except] ~x1[except]+x2[except])

summary(tlm)
plot(tlm)

install.packages("car")
install.packages("MASS")
library(MASS)

