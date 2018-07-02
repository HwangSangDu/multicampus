install.packages("RMySQL")
library(RMySQL)

con <- dbConnect(drv = MySQL(),
          dbname="iotyang",
          user="iot_yang" ,
          password="seculix77",
          host= "iotyang.cssvmqyy96ui.us-east-2.rds.amazonaws.com" )



install.packages("rvest")
library(rvest)

repair_encoding(dbListTables(conn = con))
dbListFields(con , 고객)
dbGetQuery(com, "select * from ~")

table1 = paste("select * from", tables[1])
str(sakila_film)

dbDisconnect(con)

# database 비밀번호 변경!!!
# ALTER USER 'root'@'localhost' IDENTIFIED^
# WITH mysql_native_password BY '1234'




# var myhost = 'a1mdaxda15mrnu.iot.us-west-2.amazonaws.com';
# var dbhost = 'iotyang.cssvmqyy96ui.us-east-2.rds.amazonaws.com';
# var databaseName = 'iotyang';
# var databaseID = 'iot_yang';
# var databasePW = 'seculix77';


install.packages("vcd")
library(vcd)
counts<-table(Arthritis$Improved)
barplot(counts, 
        main = 'Simple',
        xlab = "imporve",
        ylab = "Frequency")

counts






# 사진 저장하기
?pdf
?jpeg
png(
  filename = "example.png"
  # height = 
  # width = 
)
barplot(counts, 
        main = 'Simple',
        xlab = "imporve",
        ylab = "Frequency",
        horiz = T)

?barplot
dev.off()



plot(Arthritis$Improved,
     main = 'Simple',
     xlab = "imporve",
     ylab = "Frequency",
     horiz = T)


methods("plot")
counts1 = table(Arthritis$Improved,
      Arthritis$Treatment)
barplot(counts1,
        xlab = "Treatement",
        ylab = "frequency",
        col = c("snow4","yellow2","violet"),
        legend = rownames(counts1),
        beside = T) # 묶은 막대형
colors()
?barplot

# 스피노 그램
spine(counts1)
spine(t(counts1))


counts1
t(counts1)

counts
par(mfrow=c(1,1))
par(mfrow=c(1,2))
pie(counts)


?pie

install.packages("plotrix")
library(plotrix)

pie3D(counts, labels = rownames(counts))
counts




# 히스토그램

mtcars$mpg
hist(mtcars$mpg, breaks=12, freq=F)
?hist

lines(density(mtcars$mpg), col = "blue", lwd = 2)
str(factor(4,5,6))

install.packages("sm")
library(sm)
attach(mtcars)


cyl.f <- factor(cyl, 
                levels=c(4,6,8),
                labels = c("4","6","8"))

sm.density.compare(mtcars$mpg,
                   mtcars$cyl,
                   xlab = "x",
                   ylab="y")


# boxplot 
stat = boxplot(mtcars$mpg~mtcars$cyl, main="car",
        data=mtcars,
        xlab="N",
        ylab="M")
# 이상치 출력
stat
stat$out


## 중앙값
mtcars$mpg
median(mtcars$mpg)
median(c(1,2,3,4))
stat$out





mtcars

dotchart(mtcars$mpg,
         labels=rownames(mtcars),
         cex =.6
         )


myvar <- c("mpg", "hp", "wt")
# summary는 중앙값 평균 최솟값 등등 전체적으로 구해준다.
summary(mtcars[myvar])
mtcars[myvar]



?summary







# 분산 , 표준편차
mtcars$mpg
var(mtcars$mpg)
sd(mtcars$mpg)
boxplot(mtcars$mpg)
hist(mtcars$mpg)


mtcars$cyl
var(mtcars$cyl)
sd(mtcars$cyl)
boxplot(mtcars$cyl)
hist(mtcars$cyl)

mtcars$disp
var(mtcars$disp)
sd(mtcars$disp)
hist(mtcars$disp)






