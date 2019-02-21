install.packages("VIM")
library(VIM)
# 결측값 비율 빈도를 보여준다,
?aggr
aggr(sleep, prop=F, numbers=T)
aggr(sleep, prop=T, numbers=T)
# aggr(sleep, prop=T, numbers=F)


View(sleep)
marginplot(sleep[c("Gest", "Dream")],
       pch = c(20),
       col=c("darkgray", "red", "blue"))


?marginplot 
marginplot(sleep[c("Gest", "Dream")], 
           pch = c(33),# ascii code
           col=c("black", "red", "blue")) 


?matrixplot
matrixplot(sleep)



(x <- data.frame(abs(is.na(sleep))))

















