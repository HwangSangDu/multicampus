# # 
# # 
# # 
# # ### 네이버 뉴스 크롤링 ###
# # ##########################
# # 
# # 
# # 
# # # 1. 메인 페이지 뉴스 본문 접근
# # # 태그(node) , 속성, 속성 값
# # 
# # # 2. 기사 본문 추출
# # 
# # 
# # 
# # 
# # # 크롤링 주소
# # # 페이지 번호 여부 GET 통신
# # # https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20180706&page=1
# # # https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=100&oid=421&aid=0003468279
# # # https://news.naver.com/main/read.nhn?mode=LSD&mid=sec&sid1=101&oid=123&aid=0002188000
# # # https://namu.wiki/w/%EC%9E%A5%EC%9E%90%EC%97%B0%20%EC%9E%90%EC%82%B4%20%EC%82%AC%EA%B1%B4
# # 
# # 
# # 
# # url <- "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20180706&page=1"
# # # http + r
# # install.packages("httr")
# # # harvers + r
# # install.packages("rvest")
# # 
# # library(httr)
# # library(rvest)
# # 
# # 
# # ?GET()
# # # 헤더 , 바디 정보 가져옴
# # http_naver <- GET(url)
# # content(http_naver)
# # # 메타 데이터만 가져옴
# # html_naver <- read_html(http_naver)
# # 
# # # 함수명 : html_nodes(url, css, xpath)
# # # HTML 요소 추출
# # ?html_nodes
# # html_nodes(html_naver,"div.list_body a")
# # links_area <- html_nodes(html_naver, "div.list_body a") # div 태그 안에 class list_body 안에서도 a태그 가져오겠다.
# # links_area2 <-  unique(html_attr(links_area, "href")) # 반복되는 url은 안녕
# # # grep 명령어는 해당 문자열 검색 명령어이다.
# # links_area3 <- grep("news.naver.com", links_area2, value = T) # naver뉴스만 가져오겠다
# # # Extract attributes, text and tag name from html. 
# # ?html_text
# # 
# # 
# # total_article <- c(0)
# # for (i in 1:length(links_area3)) {
# #   http_news <- GET(links_area3[i]) # 링크 하나만 선택해서 GET 요청
# #   html_news <- read_html(http_news) 
# #   contenst_area <- html_nodes(html_news, "div#articleBodyContents") # 기사 본문
# #   article <- html_text(contenst_area) # text 추출 
# #   total_article <- c(article, total_article) 
# #   Sys.sleep(2) # 트래픽 과부하 방지
# #               # 사이트에서 차단 시킨다.
# # }
# # ?grep
# # total_article <- gsub(pattern="\n", "",total_article,fixed = T)
# # total_article <- gsub(pattern="\t", "",total_article,fixed = T)
# # total_article
# # 
# # 
# # ?gsub
# # ?gsub
# # # \\\n
# # # \\{\\}
# # 
# # 
# # 
# # url <- "https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=001&date=20180706&page=1"
# # ?paste
# # # page번호 매기기
# # paste0(url , 5)
# # paste (url, "5", sep="")
# # 
# # for(i in 1:9)
# # {
# #   print(paste0(url,i))
# # }
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# # 
# 
# 
# 
# 
# 
# # ---------------------------------------------------
# # ---------------------------------------------------
# # ---------------------------------------------------
# # ---------------------------------------------------
# # ---------------------------------------------------
# # ---------------------------------------------------
# # ---------------------------------------------------
# 
# df_1 <- data.frame(cus_ids = c("A001",
#                                "A002",
#                                "A003"),
#                    name = c("홍길동",
#                             "이순신",
#                             "김철수"))
# 
# df_2 <- data.frame(cus_id = c("A001",
#                               "A002",
#                               "A003"),
#                    buy = c("책",
#                            "A4용지",
#                            "지우개"))
# 
# merge(df_1, df_2, by.x = "cus_ids", 
#       by.y = "cus_id")
# ## 만약 기준이되는 변수명이 cus_id로 같은 경우##
# merge(df_1, df_2, by = "cus_id")
# 
# 
# #### 네이버 뉴스속보 크롤링 ######
# 
# install.packages("httr")
# install.packages("rvest")
# library(httr)
# library(rvest)
# 
# total_article <-c()
# url <-"https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=101&date=20180706&page="
# for(j in 1:10){
#   base_url <- paste0(url, j)
#   print(j)  ## 반복시 페이지 번호 확인
#   # http  + r = httr
#   # harvest + r = rvest
#   http_naver <- GET(base_url)
#   html_naver <- read_html(http_naver) 
#   
#   ## class = ., id = # 
#   links_area <- html_nodes(html_naver, "div.list_body a") 
#   links_area2 <- unique(html_attr(links_area, "href"))
#   links_area3 <- grep("news.naver.com", links_area2, value = T)
#   
#   for(i in 1:length(links_area3)){
#     http_news <- GET(links_area3[i])
#     html_news <- read_html(http_news)
#     contents_area <- html_nodes(html_news, "div#articleBodyContents")
#     article <- html_text(contents_area)
#     total_article <- c(article, total_article)
#   }
#   # Sys.sleep(1)  
# }
# 
# ## 정규표현식은 패턴을 찾아주는 것
# ## 목표: 치환\
# set.seed(1234)
# test <- total_article[sample(200,50)]
# test
# clz_data <- gsub(pattern = "^.+\\{\\}",replacement = " ",x = test)
# clz_data <- gsub("[0-9a-zA-Z]",   " ",    clz_data)
# clz_data <- gsub("[[:punct:]▶◆∙ⓒ]",   " ", clz_data)
# clz_data <- gsub("[[:space:]]+", " ", clz_data)
# clz_data
# # ---------------------------------------------------
# # ---------------------------------------------------
# # ---------------------------------------------------
# # ---------------------------------------------------
# # ---------------------------------------------------
# # ---------------------------------------------------
# # ---------------------------------------------------
# 
# 
# 
# ### 정제 방법 ###
# #################
# 
# # 1.패키지
# # 2.정규표현식
# install.packages("stringr")
# 
# 
# (strings <- c("^ab", "ab", "abc", "abd", "abe", "ab 12"))
# grep("ab.", strings, value =T)
# ?gsub
# ??grep
# ?regexpr
# 
# ## 정규표현식
# # . : fixed
# # ignore.case =T 대소문자  무시
# # grepl은 TRUE FALSE 로 변환
# # perl : 몇개나 중복되는가?
# 
# # gsub 함수는 정규표현식 replace 버전이다.
# 
# # 정제의 마지막 단계는 띄어쓰기 여러 개를 1개로 바꿔주는 것이다.
# 
# ## 단어 추출 ###'
# ################
# library(KoNLP)
# 
# # KoNLP내에는 안에 자기만의 사전을 가지고 있다.
# # 단어 사전의 개수가 28만개였는데 90만개까지 개수를 증가시켰다.
# # Checking user defined dictionary! (아직 R에서는 update가 되지 않았다는 경고메세지)
# 
# 
# # 라이브러리 경로 확인 방법
# .libPaths()
# 
# 
# text <- "사과와 바나나는 맛있다"
# extra<- extractNoun(text)
# extra
# 
# library(stringr)
# 
# # 캐릭터 타입으로 캐스팅한다.
# ?as.character
# 
# ko_word <- function(x){
#   d <- as.character(x)
#   pos <- paste(SimplePos22(d))
#   # NC는 명사를 의미한다
#   str_match(string=pos ,pattern="[가-힣]+/NC") # 한글 한글자
#   extract <- str_match(string=pos ,pattern="([가-힣]+)/NC") # 여러 글자capturing
#   keyword <- extract[,2] # 2열
#   keyword[!is.na(keyword)] # NA(Not a Number) filtering
# }
# 
# # 배열을 벡터로 변경
# # pos <- paste(SimplePos22(text))
# #[가-힣]은 한글집합
# 
# # str_match(string=pos ,pattern="[가-힣]+/NC") # 한글 한글자
# # extract <- str_match(string=pos ,pattern="([가-힣]+)/NC") # 여러 글자capturing
# # keyword <- extract[,2] # 2열
# # keyword[!is.na(keyword)] # NA(Not a Number) filtering
# 
# 
# ko_word(text)
# 
# 
# 
# 
# # tm package를 사용한다.
# # term(용어)
# install.packages('tm')
# library(tm)
# ?VectorSource()
# getSources()
# clz_data
# 
# 
# ?VCorpus
# # 다른 형식으로 변경
# cps <- VCorpus(VectorSource(clz_data))
# 
# 
# # stopwords = column, cps = row 
# ?DocumentTermMatrix
# dtm <- DocumentTermMatrix(cps, control = list(
#   tokenize = ko_word,
#   stopwords = c("바로가기", "뉴스", "기자"),
#   wordLengths = c(2,7)
# ))
# 
# 
# 
# dtm.mat <- as.matrix(dtm)
# dtm.mat
# 
# order(colSums(dtm.mat), decreasing = T)
# sort(colSums(dtm.mat), decreasing = T)
# ?sort
# 
# 
# word.freq <- colSums(dtm.mat)
# 
# word.order = sort(word.freq, decreasing =T)[1:50]
# 
# df_word <- data.frame(word = names(word.order),
#                       freq = word.order)
# df_word
# 
# # install.packages("htmlwidgets")
# # library(htmlwidgets)
# # library(wordcloud2)
# wordcloud2(data=df_word ,
#            color = "orange")
# 
# # install.packages("wordcloud2")
# 
# 
# 
# dtm <- DocumentTermMatrix(cps, control = list(
#   tokenize = ko_word,
#   stopwords = c("바로가기", "뉴스", "기자"),
#   wordLengths = c(2,7),
#   weighting = weightBin
# ))
# dtm
# dtm.mat2 <- as.matrix(dtm)
# dtm.mat2
# word.freq2 <- colSums(dtm.mat2)
# 
# word.order = order(word.freq, decreasing =T)[1:50]
# 
# df_word <- data.frame(word = names(word.order),
#                       freq = word.order)
# df_word
# 
# wordcloud2(data=df_word ,
#            color = "orange")
# 
# word_50th <- dtm.mat2[1:50]
# word_50th
# occur<- t(word_50th)
# occur <- t(word_50th) %*% word_50th
# dim(occur)
# 
# 
# 
# repair_encoding(colnames())
# install.packages("qgraph")
# library(qgraph)
# ?qgraph
# qgraph(occur,
#        layout = "spring",
#        color = "skyblue",
#        vsize = sqrt(diag(occur)),
#        labels = colnames(occur))
# 
# 
# ?ggplot2
# 
# 
# png(result.png,
#     width)
# 
# qgraph
# 
# 
# 




df_1 <- data.frame(cus_ids = c("A001",
                               "A002",
                               "A003"),
                   name = c("홍길동",
                            "이순신",
                            "김철수"))

df_2 <- data.frame(cus_id = c("A001",
                              "A002",
                              "A003"),
                   buy = c("책",
                           "A4용지",
                           "지우개"))

merge(df_1, df_2, by.x = "cus_ids", 
      by.y = "cus_id")
## 만약 기준이되는 변수명이 cus_id로 같은 경우##
merge(df_1, df_2, by = "cus_id")



#### 네이버 뉴스속보 크롤링 ######

install.packages("httr")
install.packages("rvest")
library(httr)
library(rvest)

total_article <-c()
url <-"https://news.naver.com/main/list.nhn?mode=LSD&mid=sec&sid1=101&date=20180706&page="
for(j in 1:10){
  base_url <- paste0(url, j)
  print(j)  ## 반복시 페이지 번호 확인
  # http  + r = httr
  # harvest + r = rvest
  http_naver <- GET(base_url)
  html_naver <- read_html(http_naver) 
  
  ## class = ., id = # 
  links_area <- html_nodes(html_naver, "div.list_body a") 
  links_area2 <- unique(html_attr(links_area, "href"))
  links_area3 <- grep("news.naver.com", links_area2, value = T)
  
  for(i in 1:length(links_area3)){
    http_news <- GET(links_area3[i])
    html_news <- read_html(http_news)
    contents_area <- html_nodes(html_news, "div#articleBodyContents")
    article <- html_text(contents_area)
    total_article <- c(article, total_article)
  }
  Sys.sleep(2)  
}

## 정규표현식은 패턴을 찾아주는 것
## 목표: 치환\
set.seed(1234)
test <- total_article[sample(200,3)]

clz_data <- gsub("^.+\\{\\}", " ", total_article)
clz_data <- gsub("[0-9a-zA-Z]",   " ",    clz_data)
clz_data <- gsub("[[:punct:]▶◆∙ⓒ]",   " ", clz_data)
clz_data <- gsub("[[:space:]]+", " ", clz_data)

## 정제 결과 확인
set.seed(5678)
clz_data[sample(200,3)]
clz_data


## 단어 추출
library(KoNLP)
useNIADic()

## 패키지 저장 경로
.libPaths()

### 단어추출 함수 정의 ###
text <- "사과와 바나나는 맛있다."
extractNoun(text)

SimplePos09(text)

library(stringr)

ko_word <- function(x){
  d <- as.character(x)
  pos <- paste(SimplePos22(d))
  extract <- str_match(pos, "([가-힣]+)/NC")
  keyword <- extract[,2]
  keyword[!is.na(keyword)]
}

## 텍스트 마이닝 #####
install.packages("tm")
library(tm)
getSources()

cps <- VCorpus(VectorSource(clz_data))
dtm <- DocumentTermMatrix(cps, control = list(
  tokenize = ko_word,
  stopwords = c("바로가기", "뉴스", "기자"),
  wordLengths = c(2 , 7)
))

dtm.mat<- as.matrix(dtm)
word.freq <- colSums(dtm.mat)
df_word <- data.frame(word = names(word.freq),
                      freq = word.freq)


#### 워드클라우드 그리기 ####
install.packages("wordcloud2")
install.packages("htmlwidgets")
library(wordcloud2)
library(htmlwidgets)

wc2 <- wordcloud2(data = df_word,
                  color = "random-dark")
saveWidget(wc2, "tmp.html", selfcontained = F)


dtm2 <- DocumentTermMatrix(cps, control = list(
  tokenize = ko_word,
  stopwords = c("바로가기", "뉴스", "기자"),
  wordLengths = c(2 , 7),
  weighting = weightBin,
  dictionary= c("무단전재", "재배포")
))

dtm.mat2<- as.matrix(dtm2)
word.freq2 <- colSums(dtm.mat2)
word.order2<- order(word.freq2, decreasing = T)[1:50]

word_50th <- dtm.mat2[, word.order2]

occur <- t(word_50th) %*% word_50th
dim(occur)

##### 연관 그래프 그리기 #####
install.packages("qgraph")
library(qgraph)

repair_name <-repair_encoding(colnames(occur))

png('result.png', 
    width = 1500,
    height = 1500)
qgraph(occur,
       layout = "spring",
       color = "skyblue",
       vsize = sqrt(diag(occur))/2,
       labels = repair_name,
       diag = F
)
dev.off()

#### dictionary 사용해서 dtm 매트릭스 만들기 ####

dtm2 <- DocumentTermMatrix(cps, control = list(
  tokenize = ko_word,
  stopwords = c("바로가기", "뉴스", "기자"),
  wordLengths = c(2 , 7),
  weighting = weightBin,
  dictionary= c("무단전재", "재배포")
))

dtm.mat2<- as.matrix(dtm2)







ls()
rm(list = ls())





























