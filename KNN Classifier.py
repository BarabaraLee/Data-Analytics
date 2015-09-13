import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors

#------------------------------------Sample data generation
mean1_5=[1,0]
mean6_10=[0,1]
cov=[[1,0],[0,1]]
cov_sub=[[0.2,0],[0,0.2]]

x1_5,y1_5=np.random.multivariate_normal(mean1_5,cov,5).T
x6_10,y6_10=np.random.multivariate_normal(mean6_10,cov,5).T

m_x=np.concatenate((x1_5,x6_10))
m_y=np.concatenate((y1_5,y6_10))
#plt.plot(m_x,m_y,'x',color ='r',marker='D')

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
Tpoints=np.c_[point_x,point_y]

#------------------------------------K-nearest neighbor classifier
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


print performance(classlabel,KNNDecisions[0])
#(0.017857142857142856, 0.06818181818181818)
print performance(classlabel,KNNDecisions[1])
#(0.0, 0.0)
