#Prob 3
spam.dat<-read.table('spam.data.txt',sep=' ')
spam.dat$V58<-as.factor(spam.dat$V58)
#Part a
tree.model1<-tree(V58~.,data=spam.dat,mindev=0.0008)
cv.model <- cv.tree(tree.model1)
plot(cv.model)
size=110
size110tree<-prune.tree(tree.model1, best=size)
plot(size110tree)
plot(size110tree)
text(size110tree,cex=0.5)
summary(size110tree)
# Misclassification error rate: 0.03478 = 160 / 4600 

#Part b
tree.model2<-tree(V58~.,data=spam.dat,mindev=0.005)
cv.model <- cv.tree(tree.model2)
plot(cv.model)
size=20
size20tree<-prune.tree(tree.model2, best=size)
plot(size20tree)
plot(size20tree)
text(size20tree,cex=0.5)
summary(size20tree)
# Misclassification error rate: 0.07826 = 360 / 4600

#Part c
tree.model3<-tree(V58~.,data=spam.dat,mindev=0.006)
cv.model <- cv.tree(tree.model3)
plot(cv.model)
best.size <- cv.model$size[which(cv.model$dev==min(cv.model$dev))] 
best.size #[1] 18 17 16 15 14 13
bestsizetree <- prune.tree(tree.model3, best=best.size)
plot(bestsizetree)
text(bestsizetree,cex=0.8)
summary(bestsizetree)#number of terminal nodes is 13.
# Misclassification error rate: 0.08261 = 380 / 4600 







