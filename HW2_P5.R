#Part5

spam.dat<-read.table('spam.data.txt',sep=' ')
spam.dat$V58<-as.factor(spam.dat$V58)