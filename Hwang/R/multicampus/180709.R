# 역행렬 공식



# mat <- matrix(1:4,nrow=2)
# 역행렬 determinant
?solve()

search()
getwd()

# type check
typeof("")
class("")
rm(list = ls())

# as.data.frame
# as는 객체를 데이터 프레임으로 변경


factor("f",c("m","f"))


seq_along(c('a','b','c','d'))
seq_len(3)


5 < 5
5 < 6

T + 1
F + 1

x <- NULL

# 정수 정수, 실수, 복소수, 부울
# 문자


x <- 1:100
y <- replace(x,c(1,2,3),c(33,22,11))
z <- append(x,y, after = 10)
z
seq(from=1, to=100, 0.001)
?cbind # 데이터 프레임을 갖다 붙인다.


?sample



seed(1234)
x <- runif(5)
x
any(x > 0.5)
all(x < 0.7)
is.vector(x)




matrix(1:9, nrow=3)
vec1 <- c(1,2,3)
vec2 <- c(4,5,6)
vec3 <- c(7,8,9)
mat1 <- rbind(vec1, vec2, vec3)
mat1
mat1 <- cbind(vec1, vec2, vec3) 
mat1


?apply
apply(mat1)
x <- cbind(x1 = 3, x2 = c(4:1, 2:5))
x
dimnames(x)[[1]] <- letters[1:8]
apply(x, 2, mean, trim = .2)
x
col.sums <- apply(x, 2, sum) # 2번은 열 기준 1번은 행 기준
row.sums <- apply(x, 1, function(x){
  return (x)
})
col.sums
row.sums


?data.frame()

L3 <- LETTERS[1:3]
L3
fac <- sample(L3, 10, replace = TRUE)
fac
(d <- data.frame(x = 1, y = 1:10, fac = fac))
d$y

## The "same" with automatic column names:
data.frame(1, 1:10, sample(L3, 10, replace = TRUE))

is.data.frame(d)

## do not convert to factor, using I() :
(dd <- cbind(d, char = I(letters[1:10])))
rbind(class = sapply(dd, class), mode = sapply(dd, mode))

stopifnot(1:10 == row.names(d))  # {coercion}

(d0  <- d[, FALSE])   # data frame with 0 columns and 10 rows
(d.0 <- d[FALSE, ])   # <0 rows> data frame  (3 named cols)
(d00 <- d0[FALSE, ])  # data frame with 0 columns and 0 rows




# data 프레임
# list = [키-값] = 딕셔너리
(ff <- factor(substring("statistics", 1:10, 1:10), levels = letters))
as.integer(ff)      # the internal codes
ff
(f. <- factor(ff))  # drops the levels that do not occur
ff[, drop = TRUE]   # the same, more transparently



x <- c(1,2,3)
x


?sink()
mtcars
summary(mtcars)
str(mtcars)
dim(mtcars)

temp1 <- mtcars[1:2,]
temp2 <- mtcars[3:6,]
temp1
temp2
new <- rbind(temp1,temp2)
new


data <- scan()
data
?scan()

install.packages("xlsx")


rm (list = ls())
manager <- c(1,2,3,4,5)
date <- c("10/24/08", "10/28/08", "10/1/08", "10/12/08", "5/1/9")
country <- c("US", "US", "UK", "UK", "UK")
gender <- c("M", "F", "F", "M", "F")
age<- c(32,12,32,12,99)
q1 <- c(5,3,3,3,2)
q2 <- c(4,5,5,3,2)
q3 <- c(5,2,5,4,1)
q4 <- c(5,5,5,1,2)
q5 <- c(5,5,2,1,1)
leadership <- data.frame(manager,date,country,gender,age,q1,q2,q3,q4,q5, stringsAsFactors = F)
leadership


na.omit(leadership)
as.Date("01/01/01","%m/%d/%y")
Sys.Date()

?merge
?merge()


iris



summary(iris)



colnames(iris)
rownames(iris)




?subset

subset(airquality, Temp < 60, select = -Temp)





