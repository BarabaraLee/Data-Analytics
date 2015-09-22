import numpy as np
import matplotlib.pyplot as plt
from itertools import product

#Problem 1
#------------------------------------Part a
mean1_5=[1,0]
mean6_10=[0,1]
cov=[[1,0],[0,1]]
cov_sub=[[0.2,0],[0,0.2]]

x1_5,y1_5=np.random.multivariate_normal(mean1_5,cov,5).T
x6_10,y6_10=np.random.multivariate_normal(mean6_10,cov,5).T

m_x=np.concatenate((x1_5,x6_10))
m_y=np.concatenate((y1_5,y6_10))
plt.plot(m_x,m_y,'x',color ='r',marker='D')

point_x=[]
point_y=[]
classlabel=[]
subclasslabel=[]
for count in range(100):
    i=np.random.random_integers(1,10)
    subclasslabel.append(i)
    if i in range(1,6): 
        classlabel.append(1)
    else:
        classlabel.append(2)
    x,y=np.random.multivariate_normal([m_x[i-1],m_y[i-1]],cov_sub,1).T
    point_x.append(x[0])
    point_y.append(y[0])
for i in range(100):
    if classlabel[i]==1: plt.plot(point_x[i],point_y[i],'o',c='blue')
    elif classlabel[i]==2: plt.plot(point_x[i],point_y[i],'o',c='orange')
print point_x
print point_y
print classlabel
print subclasslabel
#------------------------------------Part b
z=map(lambda x:x-1,classlabel)
ones=np.ones(100)
X=np.mat(np.column_stack((ones,point_x,point_y)))

beta=(X.T*X).I*X.T*np.mat(z).T
X*beta

coordinates = list(product(xrange(-30,40), xrange(-30,40)))
coor=map(lambda (x,y):(x/10.0,y/10.0),coordinates)
beta=np.array(beta)

for i in range(4900):
    if (beta[0]+beta[1]*coor[i][0]+beta[2]*coor[i][1])[0]>0.5: 
        plt.plot(coor[i][0],coor[i][1],'x',c='orange')
    elif (beta[0]+beta[1]*coor[i][0]+beta[2]*coor[i][1])[0]<=0.5: 
        plt.plot(coor[i][0],coor[i][1],'x',c='blue')
        
Decision=[]
for i in range(100):
    if (beta[0]+beta[1]*point_x[i]+beta[2]*point_y[i])[0]>0.5: 
        Decision.append(2)
    elif (beta[0]+beta[1]*point_x[i]+beta[2]*point_y[i])[0]<=0.5: 
        Decision.append(1)

x=range(-30,40)
x=map(lambda x:x/10.0,x)
plt.plot(x,(0.5-beta[0]-x*beta[1])/beta[2],color='black')
plt.ylim([-3,4])
plt.show()

Decision=map(lambda x: x-1, Decision)
classlabel=map(lambda x: x-1, classlabel)

def performance(y_actual, y_hat):
    FP = 0
    FN = 0
    TP = 0
    TN = 0
    for i in range(len(y_hat)): 
        if y_hat[i]==1 and y_actual[i]==0:
           FP += 1
        if y_hat[i]==0 and y_actual[i]==1:
           FN += 1
        if y_hat[i]==1 and y_actual[i]==1:
           TP += 1   
        if y_hat[i]==0 and y_actual[i]==0:
           TN += 1
    FPR = FP*1.0/(FP+TN) 
    FNR = FN*1.0/(FN+TP) 
    return (FPR,FNR)
    
print performance(classlabel,Decision)
#(0.18, 0.14)
