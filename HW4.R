library(graphicsQC)
require(graphics)
library(graphics)
data=read.csv('/Users/linjunli/Desktop/Data Analytics/HW/HW4/cereal.txt',header=F)
rownames(data)=c('Apple Cinnamon Cheerios',
                 'Basic 4',
                 'Cheerios'
                 ,'Cinnamon Toast Crunch'
                 ,'Clusters'
                 ,'Cocoa Puffs'
                 ,'Count Chocula'
                 ,'Crispy Wheat & Raisins'
                 ,'Golden Grahams'
                 ,'Honey Nut Cheerios'
                 ,'Kix'
                 ,'Lucky Charms'
                 ,'Multi-Grain Cheerios'
                 ,'Oatmeal Raisin Crisp'
                 ,'Raisin Nut Bran'
                 ,'Total Corn Flakes'
                 ,'Total Raisin Bran'
                 ,'Total Whole Grain'
                 ,'Triples'
                 ,'Trix'
                 ,'Wheaties'
                 ,'Wheaties Honey Gold')
datadist=dist(scale(data), method = "euclidean")
loc <- cmdscale(datadist)
x <- loc[, 1]
y <- -loc[, 2]
plot(x, y, xlab = "", ylab = "", asp = 1,main = "Multidimensional Scaling",type = "p",col='red')
text(x, y, rownames(data), cex = 0.7)

#-----------PCA implimentation
mdata=as.matrix(data)
smdata=scale(mdata)
cov1=t(smdata)%*%smdata
V_trunc=svd(cov1)$v[,1:2]
LDdata=smdata%*%V_trunc
xx <- LDdata[, 1]
yy <- LDdata[, 2]
plot(xx, yy, xlab = "", ylab = "", asp = 1,main = "PCA",type = "p",col='blue')
text(xx, yy, rownames(data), cex = 0.7)

plot(-xx, -yy, xlab = "", ylab = "", asp = 1,main = "PCA Mirror",type = "p",col='green')
text(-xx, -yy, rownames(data), cex = 0.7)
