# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:06:01 2021

@author: cui
"""
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn import  svm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def create_data():
    iris=load_iris()
    df=pd.DataFrame(iris.data,columns=iris.feature_names)
    df['label']=iris.target
    df.columns=['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data=np.array(df.iloc[:100,[0,1,-1]])
    #对切片好的数据进行标准化处理
    sc=MinMaxScaler()
    sc.fit(data)
    data=sc.transform(data)
    data[:,-1]=data[:,-1]*2-1
    #划分测试样本
    test=np.vstack((data[:15],data[50:65]))    
    #10个有标签的样本
    label_sampled=np.vstack((data[15:20],data[65:70]))
    #60个无标签样本
    unlabel_sampled=np.vstack((data[20:50],data[70:]))
    return test,label_sampled,unlabel_sampled

test,labeled,unlabeled=create_data()
clf=svm.SVC(C=1,kernel='linear')
#使用有标签的数据集训练
clf.fit(labeled[:,:2],labeled[:,-1])
positive_labeled=labeled[5:]
negative_labeled=labeled[:5]
plt.scatter(labeled[:5,:2][:,0], labeled[:5,:2][:,1], color='red',s=40,label=-1)
plt.scatter(labeled[5:,:2][:,0], labeled[5:,:2][:,1], color='blue',s=40,label=1)
x_points=np.linspace(0, 1,10)
y_points=-(clf.coef_[0][0]*x_points+clf.intercept_)/clf.coef_[0][1]
plt.plot(x_points, y_points,'green')
plt.legend()


#预测，伪标记
fake_label=clf.predict(unlabeled[:,:2])
unlabeled_positive_x=[]
unlabeled_positive_y=[]
unlabeled_negative_x=[]
unlabeled_negative_y=[]
for i in range(len(unlabeled)):
    if fake_label[i]==1:
        unlabeled_positive_x.append(unlabeled[i,0])
        unlabeled_positive_y.append(unlabeled[i,1])
    else:
        unlabeled_negative_x.append(unlabeled[i,0])
        unlabeled_negative_y.append(unlabeled[i,1])
plt.scatter(unlabeled_positive_x,unlabeled_positive_y,color='red',s=15)
plt.scatter(unlabeled_negative_x, unlabeled_negative_y, color='blue',s=15)   
print("经过标记的样本训练后，对被标记的样本的预测的正确率为{}".format(clf.score(unlabeled[:,:2], unlabeled[:,-1])))

#初始化Cu,Cl
Cu=0.1
Cl=1
weight=np.ones(len(labeled)+len(unlabeled))
weight[len(unlabeled):]=Cu

# 用于训练有标记与无标记样本集合
train_sample=np.vstack((labeled[:,:2],unlabeled[:,:2]))
 # 用于训练的标记集合
train_label=np.hstack((labeled[:,-1],fake_label))
unlabeled_id=np.arange(len(unlabeled))

while Cu>Cl:
    clf.fit(train_sample,train_label,sample_weight=weight)
    while True:
        #通过训练得到的预测标记
        predicted_y=clf.decision_function(unlabeled[:,:2])
        #伪标记，这里为与预测的区分开，写为real_y
        real_y=fake_label
        epsilon = 1 - predicted_y*real_y
        positive_set,positive_id=epsilon[real_y>0],unlabeled_id[real_y>0]
        negative_set,negative_id=epsilon[real_y<0],unlabeled_id[real_y<0]
        positive_max_id=positive_id[np.argmax(positive_set)]
        negative_max_id=negative_id[np.argmax(negative_set)]
        epsilon1,epsilon2=epsilon[positive_max_id],epsilon[negative_max_id]
        if epsilon1>0 and epsilon2>0 and round(epsilon1+epsilon2,3)>=2:
            fake_label[positive_max_id]=-fake_label[positive_max_id]
            fake_label[negative_max_id]=-fake_label[negative_max_id]
            train_label=np.hstack((labeled[:,-1],fake_label))
            clf.fit(train_sample,train_label,sample_weight=weight)
        else:
            break
    #更新Cu
    Cu=min(2*Cu,Cl)
    #更新样本权重
    weight[len(unlabeled):]=Cu
    
x_points=np.linspace(0,1,10)
y_points=-(clf.coef_[0][0]*x_points+clf.intercept_)/clf.coef_[0][1]
plt.plot(x_points, y_points,color='yellow')
plt.show()
print('经过TSVM训练后，对未标记样本的预测正确率为{}'.format(clf.score(unlabeled[:,:2],unlabeled[:,-1])))
print('经过TSVM训练后，对测试样本的预测正确率为{}'.format(clf.score(test[:,:2],test[:,-1])))
