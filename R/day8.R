mean.seq <- function(x){
  n <- length(x)
  sum <- 0
  n2 <-0
  
  for( i in 1:n){
    newx <- i * x[i]
    sum <- sum + newx
    n2 <- n2 + i
  }
  return(sum/n2)
}

set.seed(1234)
y1 <- rep(NA, 1000)
y2 <- rep(NA, 1000)

for( i in 1:1000){
  smp <- rnorm(3)
  y1[i] <- mean(smp)
  y2[i] <- mean.seq(smp)
}

n1 <- length(y1[(y1>-0.1)&(y1<0.1)])
n2 <- length(y2[(y2>-0.1)&(y2<0.1)])

data.frame(mean = mean(y1), 
           var = var(y1), n = n1  )
data.frame(mean = mean(y2), 
           var = var(y2), n = n2  )




?read.csv
sns.c <- read.csv("snsbyage.csv",
                  header = T,
                  stringsAsFactors = default.stringsAsFactors())
sns.c                  
str(sns.c)
?factor
sns.c <- transform(sns.c,
                   age.c = factor(age,
                                  levels = c(1,2,3),
                                  labels = 
                                    c(
                                  "20대",
                                  "30대",
                                  "40대")))

sns.c <- transform(sns.c,
                   service.c = factor(service,
                                  levels = c("F","T","K","C","E")))
                                  # labels = "20대","30대","40대")
sns.c


c.tab <- table(sns.c$age.c, sns.c$service.c) 13. (a.n <- margin.table(c.tab, margin=1))
(s.n <- margin.table(c.tab, margin=2))
(s.p <- s.n / margin.table(c.tab))
(expected <- a.n %*% t(s.p))
sns.c 

margin.table(c.tab, margin = 1)


# UCBA는 내장함수
str(UCBAdmissions)
apply(UCBAdmissions, c(1,2), sum)
    
?apply




























