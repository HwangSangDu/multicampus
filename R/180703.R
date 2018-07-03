



quantile(mtcars$mpg)

myvars <- c("")
mtcars




# 베르누이 시행


x <- c(0,1,2)
px <- c(1/4, 1/2, 1/4)
Ex <- sum(x * px)
Var <- sum(x^2 * px) - Ex^2
sum(x^2)
sum(x)
Var

cat("\014")


# 이항 분포
b <- 6
p <- 1/3

n <- 6
x <- 0:n



px <- dbinom(x = x,
       size = n,
       prob = p
)

### n이하를 더해준다.
### 누적 분포함수
?qbinom
?pbinom
px <- pbinom(q = x,
             size = n,
             prob = p
)
px



px <- qbinom(p = 0.50,
             size = n,
             prob = p
)
px

plot(x, px, typ='height')


px



px <- rbinom(
  n = 10,
  size = 100,
  prob = p
)
plot(px , type='h')


b1 <- dbinom(x = 0:5,
             size=5,
             prob= 2/3)

b1
plot(b1)

b1 <- dbinom(x = 0:15,
             size=15,
             prob= 2/3)
b1
plot(b1)

b2 <- dbinom(x = 0:30,
             size=30,
             prob= 2/3)
b2
plot(b2)

b3 <- dbinom(x = 0:60,
             size=60,
             prob= 2/3)
b3
plot(b3)

b4 <- dbinom(x = 0:6000,
             size=6000,
             prob= 2/3)
b4
plot(b4, xlim = c(0,6000),
     col='red',
     type='l')
lines(b1, col = 'blue')




### 정규분포 ###
mu <- 170 # 평균
sigma <- 6 # 표준편차
ll <- mu - 3*sigma
up <- mu + 3*sigma
x <- seq(ll, up, by = 0.01)
nd <- dnorm(x, mean=mu,
            sd = sigma)
plot(nd)
?plot
?dnorm


px <- pnorm(q = 182,
            mean = mu,
            sd = 7)
px

## 0.25를 갖게 해주는 값을 찾아준다.
qnorm(p = .25,
      mean = mu,
      sd = 5)

## random extraction
rx <- rnorm(n = 3000,
       mean = mu,
       s = 10000)
rx <- sort(rx, decreasing = F)
plot(rx)





options(digits = 13)
set.seed(1234)





qnorm(p = 0.1,
      mean = 0,
      sd = 10)


r.n = rnorm(10)
r.n
sum_ini <- 0
sum_ini
for(i in r.n)
{
  sum_ini = sum_ini + i
}
sum_ini


ma <- matrix(1:12 , nrow = 3)
for (i in 2:9) {
  for (j in 1:9) {
    # print(sprintf("%d ",j*i))
    cat (i*j," ")
  }
  cat("\n")
}




### 정규분포를 이용한 표본 집단에대한 평균과 분산
m10 <- rep(NA, 1000)
m40 <- rep(NA, 1000)
for(i in 1:1000)
{
  m10[i] <- mean(rnorm(1))
  m40[i] <- mean(rnorm(30))
}

options(digits = 4)
cat("m10 평균 : " , mean(m10), "표준편차 : ", sd(m10))
cat("m40 평균 : " , mean(m40), "표준편차 : ", sd(m40))
m10
m40

par(mfrow=c(1,1))
hist(m10, xlim= c(-1.5,1.5))
hist(m40, xlim= c(-1.5,1.5))
# v(x) : m10 < m40

n <- 1000
r.1.mean <- rep(NA, n)
r.2.mean <- rep(NA, n)
for(i in 1:1000){

  r.1.mean[i] = mean(rnorm(4, mean=3, sd=1))
  r.2.mean[i] = mean(rnorm(4, mean=170 , sd=1))

  # r.1.sd = sd(rnorm(4, mean=3, sd=1))
  # r.2.sd = sd(rnorm(4, mean=170 , sd=6))

}


sd(r.1.mean)
sd(r.2.mean)
mean(r.1.mean)
mean(r.2.mean)


hist(r.1.mean, freq = F,
     col = "red")
hist(r.2.mean, freq = F,
     col = "orange")


r.1.sd
r.2.sd
hist(r.1.mean, freq = F,
     col = "red")
x1 <- seq(min(r.1.mean),
          max(r.1.mean),
          length =1000)
x1

y1 <- dnorm(x = x1,
            mean = 3,
            sd=(1/sqrt(4)))

lines(x1, y1, lty = 2, lwd = 2, col = "blue")


hist(r.2.mean, freq = F,
     col = "red")
x1 <- seq(min(r.2.mean),
          max(r.2.mean),
          length =1000)
x1

y1 <- dnorm(x = x1,
            mean = 170,
            sd=(6/sqrt(4)))

lines(x1, y1, lty = 2, lwd = 2, col = "blue")






## 정규분포가 아닌 경우
## 표본 평균의 분포
set.seed(10)
t <- 10
p <- 0.1
x <- 0:10
n <- 1000

b.2.mean <- rep(NA, n)
b.4.mean <- rep(NA, n)
b.32.mean <- rep(NA, n)

for (i in 1:n) {
  b.2.mean[i] <- mean(rbinom(2,size = t,
              prob = p
              ))

  b.4.mean[i] <- mean(rbinom(4,size = t,
                              prob = p
  ))

  b.32.mean[i] <- mean(rbinom(32,size = t,
                              prob = p
  ))
}

b.2.mean
b.4.mean
b.32.mean


par(mfrow=c(1,3))

hist(b.2.mean)
hist(b.4.mean)
hist(b.32.mean)








cor1 = c(234, 234, 234, 233, 233,
         233, 233, 231, 232, 231)

cor2 = c(146.3, 146.4, 144.1, 146.7,
         145.2, 144.1, 143.3, 147.3,
         146.7, 147.3)

## 사용자 정의 함수
# 모집단에대한 분산
var.p <- function(x,y,z){
  n <- length(x)
  m <- mean(x)
  num <- sum((x-m)^2)
  denom <- n
  var <- num / denom  # E(x)^2 / n
  return (var)
}

var.p(cor1)
















