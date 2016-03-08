#Prob2 
#Part a
 setwd("~/Desktop/Data Analytics/HW/HW5")
install.packages('glmnet')
library(glmnet)
X_train=read.table('X_train.txt',sep='',header = FALSE)
Y_train=read.table('y_train.txt',sep='',header=FALSE)

#------------------------SVM-------------------------
install.packages('e1071')
library(e1071)

## tune `svm' for classification with linear-kernel (default in svm),
## using one split for training/validation set

Train=data.frame(cbind(X_train,Y_train))
Train$V1.1=as.factor(Train$V1.1)
obj <- tune(svm, V1.1~.,data=Train,kernel='linear',
            ranges = list(cost = c(0.02,0.05,0.06,0.07,0.1,0.13)), 
            tunecontrol = tune.control(cross=5) )
# Parameter tuning of ‘svm’:
# - sampling method: 5-fold cross validation 
# - best parameters:
#   cost
# 0.1
# - best performance: 
# 0.01455422 
# - Detailed performance results:
#   cost      error  dispersion
# 1 0.02 0.01904244 0.002095551
# 2 0.05 0.01632237 0.002258410
# 3 0.06 0.01605035 0.003213973
# 4 0.07 0.01564228 0.002679766
# 5 0.10 0.01455422 0.002482853
# 6 0.13 0.01509816 0.001949963
svmfit <- svm(V1.1~.,data=Train,kernel='linear',
            cost=0.1, cross=5 )
summary(svmfit)
# Call:
#   svm(formula = V1.1 ~ ., data = Train, kernel = "linear", cost = 0.1, cross = 5)
# Parameters:
#   SVM-Type:  C-classification 
# SVM-Kernel:  linear 
# cost:  0.1 
# gamma:  0.001782531 
# Number of Support Vectors:  903
# ( 218 223 96 117 107 142 )
# Number of Classes:  6 
# Levels: 
#   1 2 3 4 5 6
# 5-fold cross-validation on training data:
#   Total Accuracy: 98.49021 
# Single Accuracies:
#   98.63946 98.57143 98.5724 98.5034 98.16451 
# error rates
1-c(98.63946, 98.57143, 98.5724, 98.5034, 98.16451)/100
# 0.0136054 0.0142857 0.0142760 0.0149660 0.0183549
# Second edition:
# 1-c(98.77551, 98.29932, 98.5724, 98.36735, 98.5724)/100
# 0.0122449 0.0170068 0.0142760 0.0163265 0.0142760
SV=svmfit$SV
install.packages('matrixStats')
library(matrixStats)
vars=colVars(SV)
#vars[vars>1]
which(vars>1)
# [1]    1   2   3  18  23  25  28  31  32  33  36  37  39  40  44  45  46  47  48
# [20]  49  56  60  61  62  63  64  65  66  67  68  69  78  79  80  83 107 108 110
# [39] 111 112 113 114 115 116 117 119 120 121 122 123 144 148 149 152 158 159 160
# [58] 161 170 177 188 189 191 195 197 198 199 200 210 211 212 213 223 224 225 226
# [77] 236 237 238 251 262 263 264 265 280 291 294 296 301 302 322 324 327 328 336
# [96] 337 338 342 358 373 374 375 379 389 401 406 417 436 438 454 462 463 464 465
# [115]466 467 468 470 471 472 474 507 513 514 515 526 551 552
# second edition:
# [1]   1   2   3  18  23  25  28  31  32  33  36  37  39  40  44  45  46  47  48
# [20]  49  56  60  61  62  63  64  65  66  67  68  69  78  79  80  83 107 108 110
# [39] 111 112 113 114 115 116 117 119 120 121 122 123 144 148 149 152 158 159 160
# [58] 161 170 177 188 189 191 195 197 198 199 200 210 211 212 213 223 224 225 226
# [77] 236 237 238 251 262 263 264 265 280 291 294 296 301 302 322 324 327 328 336
# [96] 337 338 342 358 373 374 375 379 389 401 406 417 436 438 454 462 463 464 465
# [115] 466 467 468 470 471 472 474 507 513 514 515 526 551 552

X_test=read.table('X_test.txt',sep='',header=FALSE)
X_test=data.frame(X_test)
pred <- predict(svmfit,X_test)
pred_labels=as.matrix(as.numeric(pred))
summary(pred)
#  1   2   3   4   5   6 
# 517 469 402 454 568 537 
write.csv(pred_labels, file = "SVM.txt",row.names=FALSE)