import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fsolve
from matplotlib import cm
#------------------------------------Sample data generation
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
    elif i in range(6,11):
        classlabel.append(2)
    x,y=np.random.multivariate_normal([m_x[i-1],m_y[i-1]],cov_sub,1).T
    point_x.append(x[0])
    point_y.append(y[0])
for i in range(100):
    if classlabel[i]==1: plt.plot(point_x[i],point_y[i],'o',c='blue')
    elif classlabel[i]==2: plt.plot(point_x[i],point_y[i],'o',c='orange')
'''
#------------------------------------Nearest Center Classifier
def dist(x,y):
    distances=[]
    for i in range(10):
        distances.append(map(lambda p_x,p_y:(m_x[i]-p_x)**2+(m_y[i]-p_y)**2,x,y))
    return np.transpose(distances)

def findmaplabels(x):
    d2=100000000
    c=-1
    for i in range(len(x)):
        if x[i]<d2:
            c=i
            d2=x[i]
    return c+1

coordinates = list(product(xrange(-30,40), xrange(-30,40)))
coor=np.array(map(lambda (x,y):(x/10.0,y/10.0),coordinates))
    
distances=dist(coor[:,0],coor[:,1])
maplabels=map(lambda x:findmaplabels(x),distances)

for i in range(len(coor)):
    if maplabels[i]<=5: 
        plt.plot(coor[i,0],coor[i,1],'x',c='blue')
    elif maplabels[i]>5: 
        plt.plot(coor[i,0],coor[i,1],'x',c='orange')
plt.show()

distances_s=dist(point_x,point_y)
maplabels_s=map(lambda x:findmaplabels(x),distances_s)

Decision=[]
for i in range(len(maplabels_s)):
    if maplabels_s[i]<=5: 
        Decision.append(1)
    elif maplabels_s[i]>5: 
        Decision.append(2)

Decision=map(lambda x: x-1, Decision)
classlabel=map(lambda x: x-1, classlabel)

#------------------------------------Nearest Center Performance Analysis
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
'''
#------------------------------------Bayesian Classifier
func=lambda x,y:np.exp(-2.5*((m_x[0]-x)**2+(m_y[0]-y)**2))+np.exp(-2.5*((m_x[1]-x)**2+(m_y[1]-y)**2))+np.exp(-2.5*((m_x[2]-x)**2+(m_y[2]-y)**2))+np.exp(-2.5*((m_x[3]-x)**2+(m_y[3]-y)**2))+np.exp(-2.5*((m_x[4]-x)**2+(m_y[4]-y)**2))-(np.exp(-2.5*((m_x[5]-x)**2+(m_y[5]-y)**2))+np.exp(-2.5*((m_x[6]-x)**2+(m_y[6]-y)**2))+np.exp(-2.5*((m_x[7]-x)**2+(m_y[7]-y)**2))+np.exp(-2.5*((m_x[8]-x)**2+(m_y[8]-y)**2))+np.exp(-2.5*((m_x[9]-x)**2+(m_y[9]-y)**2)))
X = np.arange(-3, 4, 0.1)
Y = np.arange(-3, 4, 0.1)
X, Y = np.meshgrid(X, Y)
CS = plt.contour(X, Y, func(X,Y),0)
#plt.clabel(CS, inline=1, fontsize=10)

plt.show()

BDecision=[]
for i in range(len(point_x)):
    if func(point_x[i],point_y[i])>0: 
        BDecision.append(1)
    elif func(point_x[i],point_y[i])<0: 
        BDecision.append(2)
#------------------------------------Bayesian Classifier Performance Analysis
BDecision=map(lambda x: x-1, BDecision)
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
    
print performance(classlabel,BDecision)