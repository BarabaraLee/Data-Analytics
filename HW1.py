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
#------------------------------------Part b
z=map(lambda x:x-1,classlabel)
ones=np.ones(100)
X=np.mat(np.column_stack((ones,point_x,point_y)))

beta=(X.T*X).I*X.T*np.mat(z).T
X*beta

coordinates = list(product(xrange(-30,40), xrange(-30,50)))
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
#------------------------------------Least Square Classifier Performance Analysis
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
#(0.06666666666666667, 0.10909090909090909)

#Problem 3-----------------------------Bayesian Classifier
classlabel=map(lambda x: x+1, classlabel)
plt.plot(m_x,m_y,'x',color ='r',marker='D')
for i in range(100):
    if classlabel[i]==1: plt.plot(point_x[i],point_y[i],'o',c='blue')
    elif classlabel[i]==2: plt.plot(point_x[i],point_y[i],'o',c='orange')
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
classlabel=map(lambda x: x-1, classlabel)
BDecision=map(lambda x: x-1, BDecision)
print performance(classlabel,BDecision)
#(0.09433962264150944, 0.0851063829787234)

#Problem 4-----------------------------Bayesian Classifier 2
classlabel=map(lambda x: x+1, classlabel)
plt.plot(m_x,m_y,'x',color ='w',marker='D')
for i in range(100):
    if classlabel[i]==1: plt.plot(point_x[i],point_y[i],'o',c='blue')
    elif classlabel[i]==2: plt.plot(point_x[i],point_y[i],'o',c='orange')
m_x_est=np.zeros(10)
m_y_est=np.zeros(10)
for i in range(10):
    m_x_est[i]=np.mean(np.extract(map(lambda x: x==i+1,subclasslabel),point_x))
    m_y_est[i]=np.mean(np.extract(map(lambda y: y==i+1,subclasslabel),point_y))
plt.plot(m_x_est,m_y_est,'x',color ='r',marker='o')
func=lambda x,y:np.exp(-2.5*((m_x_est[0]-x)**2+(m_y_est[0]-y)**2))+np.exp(-2.5*((m_x_est[1]-x)**2+(m_y_est[1]-y)**2))+np.exp(-2.5*((m_x_est[2]-x)**2+(m_y_est[2]-y)**2))+np.exp(-2.5*((m_x_est[3]-x)**2+(m_y_est[3]-y)**2))+np.exp(-2.5*((m_x_est[4]-x)**2+(m_y_est[4]-y)**2))-(np.exp(-2.5*((m_x_est[5]-x)**2+(m_y_est[5]-y)**2))+np.exp(-2.5*((m_x_est[6]-x)**2+(m_y_est[6]-y)**2))+np.exp(-2.5*((m_x_est[7]-x)**2+(m_y_est[7]-y)**2))+np.exp(-2.5*((m_x_est[8]-x)**2+(m_y_est[8]-y)**2))+np.exp(-2.5*((m_x_est[9]-x)**2+(m_y_est[9]-y)**2)))
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
#------------------------------------Bayesian Classifier 2 Performance Analysis
BDecision=map(lambda x: x-1, BDecision)
classlabel=map(lambda x: x-1, classlabel)
print performance(classlabel,BDecision)
#(0.08888888888888889, 0.05454545454545454)

#Problem 5-----------------------------KNN Classifier 2
from matplotlib.colors import ListedColormap
from sklearn import neighbors

Tpoints=np.c_[point_x,point_y]
classlabel=map(lambda x: x+1, classlabel)
k = 10
h = 0.02

# Color map for background
cmap_light = ListedColormap([(0.6, 0.8, 1), (0.9, 0.7, 0.4)])

KNNDecisions=[]
for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(k, weights=weights)
    clf.fit(Tpoints, classlabel)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = Tpoints[:, 0].min() - 1, Tpoints[:, 0].max() + 1
    y_min, y_max = Tpoints[:, 1].min() - 1, Tpoints[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
    KNNDecisions.append( clf.predict(Tpoints) )
    
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot the training points
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("2-Class Classification (k = %i, weights = '%s')"
              % (k, weights))
    
    # Plot the subclass centers
    plt.plot(m_x,m_y,'x',color ='r',marker='D')
    
    for i in range(100):
        if classlabel[i]==1: plt.plot(point_x[i],point_y[i],'o',c='blue')
        elif classlabel[i]==2: plt.plot(point_x[i],point_y[i],'o',c='orange')
plt.show()

#------------------------------------KNN Classifier Performance Analysis
KNNDecisions=map(lambda x: x-1, KNNDecisions)
classlabel=map(lambda x: x-1, classlabel)

print performance(classlabel,KNNDecisions[0])
#(0.08888888888888889, 0.09090909090909091)
print performance(classlabel,KNNDecisions[1])
#(0.0, 0.0)

