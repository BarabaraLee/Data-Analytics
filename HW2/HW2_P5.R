#Part5
library(glmnet)
spam.dat<-read.table('spam.data.txt',sep=' ')
y<-as.factor(spam.dat$V58)
X<-model.matrix(~.,spam.dat[,1:57])
cv.myNewlogit <- cv.glmnet(X,y,alpha=1,nfolds=10,family='binomial')
plot(cv.myNewlogit,xvar="lambda")
cv.myNewlogit$lambda.min #=0.0004034505