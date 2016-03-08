library('stats')
library('EMCluster')
library('xtable')
#Prob2
x=read.table('/Users/linjunli/Desktop/Data Analytics/HW/HW3/ClusterSet1.txt')
w=rep(0,20)
for( i in 1:20){
result<-kmeans(x, i)
w[i]=result$tot.withinss/result$totss
}
plot(w,xlab='Fixed Number of Clusters (K)',ylab='w (WSS/TSS)')
result<-kmeans(x, 12)
plot(x, col = result$cluster)

#Prob3
ret <- init.EM(x, nclass = 12)
# n = 1000, p = 3, nclass = 12, flag = 0, logL = -12376.8313.
# nc: 97  94  13 103  17 105 109  96  82 110  93  81
# pi: 0.09700 0.09555 0.01145 0.10299 0.01662 0.10500 0.10900 0.09600 0.08239 0.11000 0.09300 0.08100
xx=split(x,ret$class)
wss=0
wss=wss+sum((xx$`1`[,1]-mean(xx$`1`[,1]))^2+(xx$`1`[,2]-mean(xx$`1`[,2]))^2+(xx$`1`[,3]-mean(xx$`1`[,3]))^2)
wss=wss+sum((xx$`2`[,1]-mean(xx$`2`[,1]))^2+(xx$`2`[,2]-mean(xx$`2`[,2]))^2+(xx$`2`[,3]-mean(xx$`2`[,3]))^2)
wss=wss+sum((xx$`3`[,1]-mean(xx$`3`[,1]))^2+(xx$`3`[,2]-mean(xx$`3`[,2]))^2+(xx$`3`[,3]-mean(xx$`3`[,3]))^2)
wss=wss+sum((xx$`4`[,1]-mean(xx$`4`[,1]))^2+(xx$`4`[,2]-mean(xx$`4`[,2]))^2+(xx$`4`[,3]-mean(xx$`4`[,3]))^2)
wss=wss+sum((xx$`5`[,1]-mean(xx$`5`[,1]))^2+(xx$`5`[,2]-mean(xx$`5`[,2]))^2+(xx$`5`[,3]-mean(xx$`5`[,3]))^2)
wss=wss+sum((xx$`6`[,1]-mean(xx$`6`[,1]))^2+(xx$`6`[,2]-mean(xx$`6`[,2]))^2+(xx$`6`[,3]-mean(xx$`6`[,3]))^2)
wss=wss+sum((xx$`7`[,1]-mean(xx$`7`[,1]))^2+(xx$`7`[,2]-mean(xx$`7`[,2]))^2+(xx$`7`[,3]-mean(xx$`7`[,3]))^2)
wss=wss+sum((xx$`8`[,1]-mean(xx$`8`[,1]))^2+(xx$`8`[,2]-mean(xx$`8`[,2]))^2+(xx$`8`[,3]-mean(xx$`8`[,3]))^2)
wss=wss+sum((xx$`9`[,1]-mean(xx$`9`[,1]))^2+(xx$`9`[,2]-mean(xx$`9`[,2]))^2+(xx$`9`[,3]-mean(xx$`9`[,3]))^2)
wss=wss+sum((xx$`10`[,1]-mean(xx$`10`[,1]))^2+(xx$`10`[,2]-mean(xx$`10`[,2]))^2+(xx$`10`[,3]-mean(xx$`10`[,3]))^2)
wss=wss+sum((xx$`11`[,1]-mean(xx$`11`[,1]))^2+(xx$`11`[,2]-mean(xx$`11`[,2]))^2+(xx$`11`[,3]-mean(xx$`11`[,3]))^2)
wss=wss+sum((xx$`12`[,1]-mean(xx$`12`[,1]))^2+(xx$`12`[,2]-mean(xx$`12`[,2]))^2+(xx$`12`[,3]-mean(xx$`12`[,3]))^2)
tss=sum((x[,1]-mean(x[,1]))^2+(x[,2]-mean(x[,2]))^2+(x[,3]-mean(x[,3]))^2)
w=wss*1.0/tss #=0.008116935 which is smaller than te w obtained by the K-Means Algorithm.
w
A=rbind(ret$nc,ret$pi)
xtable(A,caption = 'The estimated ')

#Prob4 hierarchical clustering
d <- dist(as.matrix(x))   # find distance matrix 
hc <- hclust(d,'ward.D')                # apply hirarchical clustering 
plot(hc,xlab='Index i (Ward\'s Method)',ylab='Hight (Within Sum of Squares)',hang=-20) 
memb <- cutree(hc, k = 9)
xxx=split(x,memb)
wss=0
wss=wss+sum((xxx$`1`[,1]-mean(xxx$`1`[,1]))^2+(xxx$`1`[,2]-mean(xxx$`1`[,2]))^2+(xxx$`1`[,3]-mean(xxx$`1`[,3]))^2)
wss=wss+sum((xxx$`2`[,1]-mean(xxx$`2`[,1]))^2+(xxx$`2`[,2]-mean(xxx$`2`[,2]))^2+(xxx$`2`[,3]-mean(xxx$`2`[,3]))^2)
wss=wss+sum((xxx$`3`[,1]-mean(xxx$`3`[,1]))^2+(xxx$`3`[,2]-mean(xxx$`3`[,2]))^2+(xxx$`3`[,3]-mean(xxx$`3`[,3]))^2)
wss=wss+sum((xxx$`4`[,1]-mean(xxx$`4`[,1]))^2+(xxx$`4`[,2]-mean(xxx$`4`[,2]))^2+(xxx$`4`[,3]-mean(xxx$`4`[,3]))^2)
wss=wss+sum((xxx$`5`[,1]-mean(xxx$`5`[,1]))^2+(xxx$`5`[,2]-mean(xxx$`5`[,2]))^2+(xxx$`5`[,3]-mean(xxx$`5`[,3]))^2)
wss=wss+sum((xxx$`6`[,1]-mean(xxx$`6`[,1]))^2+(xxx$`6`[,2]-mean(xxx$`6`[,2]))^2+(xxx$`6`[,3]-mean(xxx$`6`[,3]))^2)
wss=wss+sum((xxx$`7`[,1]-mean(xxx$`7`[,1]))^2+(xxx$`7`[,2]-mean(xxx$`7`[,2]))^2+(xxx$`7`[,3]-mean(xxx$`7`[,3]))^2)
wss=wss+sum((xxx$`8`[,1]-mean(xxx$`8`[,1]))^2+(xxx$`8`[,2]-mean(xxx$`8`[,2]))^2+(xxx$`8`[,3]-mean(xxx$`8`[,3]))^2)
wss=wss+sum((xxx$`9`[,1]-mean(xxx$`9`[,1]))^2+(xxx$`9`[,2]-mean(xxx$`9`[,2]))^2+(xxx$`9`[,3]-mean(xxx$`9`[,3]))^2)
tss=sum((x[,1]-mean(x[,1]))^2+(x[,2]-mean(x[,2]))^2+(x[,3]-mean(x[,3]))^2)
w=wss*1.0/tss 
w#=0.01152203

#Prob5
x2=read.table('/Users/linjunli/Desktop/Data Analytics/HW/HW3/ClusterSet2.txt')
d <- dist(as.matrix(x2))   # find distance matrix 
hc <- hclust(d,'ward.D') 
plot(hc,xlab='Index i (Ward\'s Method)',ylab='Hight (Within Sum of Squares)',hang=-20) 
memb <- cutree(hc, k = 10)