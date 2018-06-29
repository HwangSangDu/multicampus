#install.packages('KoNLP', dependencies = T)
# install.packages('rJava')
# dyn.load('/Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk/Contents/Home/jre/lib/server/libjvm.dylib')
# Sys.setenv(JAVA_HOME = '/Library/Java/JavaVirtualMachines/jdk1.8.0_131.jdk/Contents/Home')
# Sys.setlocale("LC_ALL", "ko_KR.UTF-8")

# require(rJava)
# .jinit()
# .jcall("java/lang/System", "S", "getProperty", "java.runtime.version")


# library(rJava)
# library(KoNLP)
# KoNLP::extractNoun('아버지가 방에 들어가신다.')
# KoNLP::SimplePos09('dkqjwldqlkjdjakjhdkajdlk 아버지')


# 1
# 1.
# 1.5
# 'asd'
# "jhkjjhljklj"
# lakjsdlajd
# TRUE  T
# FALSE F

# NULL
# NA
# # 다른 언어와의 연동을 위해서 마침표를 선호하지않는다.
# # 할당연산자 == (= , <- , <<- , ->> , %>%)
# # 


# x1 = 1
# x2 = "abc"
# print(x1)
# print(x2)
# x3=1:5
# print(x3[2:5])
# print(x3[-3])

# repeat
# rep(

# )

# seq

# x4 = c(1,5,2,3,5,3,7,8,8)
# x4[-c(1,7)]
# print(x4)

# rep(3,5)
# rep(1:3,5)
# rep(1:3,each=5,length.out=76)
# rep(c(1,4,2), c(3,2,5))
# ?rep







# 1:10
# seq(1,10) 
# seq(1,10,3)


# # 무작위 추출
# sample(1:45,5)
# # 복원 추출
# sample(1:3, 10, replace= T)
# # 비복원 추출
# sample(1:10, 10, replace= F)

# # 가중치
# ?sample
# sample(1:10, ,13,prob=c(1,1,1,1,1,1,1,1,12,3))

# View(iris3)
# edit(iris3)
# str(iris)
# length(x4)
# x4


# nrow(iris3)
# ncol(iris3)

# iris3
# names(x4)
# names(iris)[2]


# rownames(iris)[2]
# colnames(iris[2])
# paste('x', 1:5, sep='')
# mode(iris)
# class(iris)
# mean(x4 )
# ind1 = sample(1:nrow(iris) ,
#               nrow(iris)*0.7, 
#               replace=T)


# x7 = matrix(nrow=10 , ncol=5)
# x8 = matrix(1:4, nrow=16,ncol=2, byrow=T)
# x8




# dim(x8)



# a1 = array(1:60,dim=c(4,5,3))
# a1


# install.packages('readbitmap')

# setwd(경로)

library(jpeg)
img1 = readJPEG("./flower.jpg")
dim(img1)
img2 = img1
img2[,,1] = img1[,,1]*0.7
img3 = img2[50:250,10:30,]
writeJPEG(img3,"newLOGO.jpg")


img4=matrix(img1, nrow=3, byrow=T)
dim(img4)


?readJPEG
?writeJPEG

# 디폴트 NULL
# 매개변수 개수 맞춘다.
data1 = data.frame(a=1:3,
                  b=c(1.5,2.2,3.4), 
                  c=LETTERS[1:3])
data1
colnames(data1)[2] = "xx" 
data1[2,'a']
# object type
data1$a

c(x4, 4,5)
y1 = matrix(1:15, nrow=3)
y2 = matrix(16:30, nrow=3)

rbind(y1,y2)
cbind(y1,y2)

y3 = list(a=1,b=1:3,c=y1, d=data1)
y3$d




cbind(y1,3,4,5,6)
y1
rbind() # row unit
cbind() # column unit







# 1:5 + 3
# a1[ , , 2]
# read
# write()
# load()
# save











y3
y3$c[1:3]
y3[[3]][1:3]
y4=list(a=y3, b=y3)
y4[1]
y4$a$d
y4$b
y4[[1]][[3]][2,3]
y4$a$c

txt1=readLines('txt1.txt',encoding='UTF-8')
class(txt1)
str(txt1)
nchar(txt1)



x5 = c(7,6,3,5)
x5 >= 6
x5[x5>=6]


# text 마이닝

txt1
txt2 = txt1[nchar(txt1)>0]
txt2


txt3 = extractNoun(txt2)
txt3
table(txt3) # list가 아니다.
txt4 = unlink(txt3)
txt4

txt4 = unlist(txt3)
txt4


table(txt4)


install.packages("wordcloud2", dependencies = T)

library(wordcloud2)

txt5 = as.data.frame(table(txt4))
# as class로 데이터 폼을 변경한다.
txt5
wordcloud2(txt5,T,'kaiti')



















