#Part 4
library(boot)
spam.dat<-read.table('spam.data.txt',sep=' ')
spam.dat$V58<-as.factor(spam.dat$V58)
LR <- glm(V58~., family='binomial', data=spam.dat)
summary(LR)
cvLR<-cv.glm(spam.dat,LR,K=10)

#THe cross validation error is:
cvLR[[3]]#[1] 0.05848635 0.05832672