n <- length(train$comment_text)
hi <- rep(0, n)
for(i in 1:n) {
  hi[i] <- length(gregexpr('[!?=-]', as.character(train$comment_text[i]))[[1]])
}
x <- table(hi, rowSums(train[,3:8])>=1)

hi2 <- rep(0, n)
for(i in 1:n) {
  hi2[i] <- length(gregexpr('[:blank:]', as.character(train$comment_text[i]))[[1]])
}
x <- table(hi2, rowSums(train[,3:8])>=1)
t.test(x[,1],x[,2],paired=T)

hi3 <- rep(0, n)
for(i in 1:n) {
  hi3[i] <- length(gregexpr('(.)\\1{5}', as.character(train$comment_text[i]))[[1]])
}
x <- table(hi3, rowSums(train[,3:8])>=1)
t.test(x[,1],x[,2],paired=T)

sorted_words <- names(sort(table(strsplit(tolower(paste(readLines("http://www.norvig.com/big.txt"), collapse = " ")), "[^a-z]+")), decreasing = TRUE))
correct <- function(word) { c(sorted_words[ adist(word, sorted_words) <= min(adist(word, sorted_words), 2)], word)[1] }

install.packages(c('hunspell','tidyverse','magrittr','text2vec','tokenizers',
                   'xgboost','glmnet','doParallel','mlr'))

library(tidyverse)
library(magrittr)
library(text2vec)
library(tokenizers)
library(xgboost)
library(glmnet)
library(doParallel)