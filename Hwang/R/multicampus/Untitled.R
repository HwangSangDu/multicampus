rm(list=ls())

# install.packages("flexclust")
library(flexclust)
data(nutrient, package = "flexclust")
# View(nutrient)

# 소문자로 변경
row.names(nutrient) <- tolower(row.names(nutrient)) 

# 표준화 (평균 : 0 , 표준편차 : 1)
nutrient.scaled <- scale(nutrient)

# 유클리드 거리 계산
?dist
(d<- dist(nutrient.scaled))


?hclust
# 매개변수 : 
# 1. dist(데이터) # dist는 유클리드 거리
# 2. method : 군집화 방법 # “single” , “complete”, “average”, complete”,“centroid”, “ward”
fit.average <- hclust(d, method = "average")


# 시각화
?plot
plot(fit.average, hang=1, cex=.8, main="Average Linkage Clustering")

# 군집 수 결정
install.packages("NbClust")
library(NbClust)
?devAskNewPage
# 새로운 페이지 출력을 시작하기 전에 
# 사용자에게 프롬프트가 표시되는지 (현재 장치에 대해) 제어하는 데 사용할 수 있습니다.
devAskNewPage(ask=TRUE)

?NbClust
# 군집분석에서 최선의 군집수와 각 군집화 인덱스 반환
nc <- NbClust(nutrient.scaled, distance = "euclidean", #유클리드
                min.nc = 2, max.nc = 15, method = "average") # 평균 (계층적)
nc
table(nc$Best, n[1,])
barplot(table(nc$Best.nc[1,]),
xlab="Number of Clusters", ylab="Number of Criteria",
main="Number of clusters chosen by 26 Criteria")

?aggregate

library(ggplot2)

?geom_jitter


install.packages("RSNNS")
library(RSNNS)
?RSNNS



x <- c(0, 0, 1, 1, 1, 1)
y <- c(1, 0, 1, 1, 0, 1)
dist(rbind(x, y))

