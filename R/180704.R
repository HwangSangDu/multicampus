# 180703

# 1교시 : 복습



#####################
#### 점추정 ####
# 점추정과 구간 추정의 차이
# http://math7.tistory.com/62?category=471451


# 함수명 : mean.seq
# 매개변수 :
# 1. x (벡터)
# 기능 : 

mean.seq <- function(x){
  n <- length(x)
  n2 <- 0
  sum <- 0
  # 길이 만큼 반복
  for (i in 1:n) {
    newx <- i*x[i] 
    sum <- sum + newx # 1*x1 + 2*x2 + 3*x3 .... (분모)
    n2 <- n2 + i # 1 + 2 + 3 + 4 .... (분자)
  }
  return (sum / n2)
}


# value init
n <- 1000
y1 <- rep(NA, n) 
y2 <- rep(NA, n)


# y1[] : sample 평균
# y2[] : 
for (i in 1:n) {
  smp <- rnorm(3) # 3개의 랜덤 샘플
  y1[i] <- mean(smp) # 표본 평균
  y2[i] <- mean.seq(smp)
}

# -0.1 ~ 0.1 extraction을 n1,n2에 대입
n1 <- length(y1[(y1 > -0.1) & (y1 < 0.1) ])
n2 <- length(y2[(y2 > -0.1) & (y2 < 0.1) ])

n1
n2
y1
y2
var(y1)

# 평균, 
data.frame(mean = mean(y1), # y1의 평균
           var = var(y1),   # y1의 분산
           n = n1)          # -0.1~0.1

data.frame(mean = mean(y2), 
           var = var(y2),
           n = n2)














#### 구간추정 #####


# 변수 초기화 (난수)
set.seed(9)
n <- 10
x <- 1:100
?seq(x, by= 0.01)
smps <- matrix(rnorm(n*length(x)), col=n) # n행 행렬



# 벡터의 단위의 연산을 가능토록 함
# apply(X, MARGIN, FUN, ...)
# 매개변수
# 1. matrix
# 2. margin(1 or 2) 열과 행의 선택  1. 열 2.행
# 3. function
# 4. 

?apply

xbar <- apply(smps, 1, mean)
se <- 1 /sqrt(10) # 분산 = 10
alpha <- 0.05     # 1 - 0.05 = 0.95 = 95% 신뢰구간 
z <- qnorm(1- alpha/2) # 0.05/2 = 0.025인 값 반환
ll <- xbar - z*se
as <- xbar + z*se
 
# 이해 안되면 http://math7.tistory.com/65?category=471451 참조

# 1. 분산을 모르는 경우 t분포를 사용하고
# 2. 분산을 아는 경우 Z분포를 사용한다.
# 3. 그러나 n(표본)이 30이상이면 신뢰할만하므로 Z분포를 사용한다고 한다(중심 극한 정리)














