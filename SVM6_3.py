# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 16:26:27 2021

@author: cui
"""
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

cancer=load_iris()
x_train,x_test,y_train,y_test=train_test_split(cancer.data, cancer.target,stratify=cancer.target,random_state=42)

#线性核函数支持向量机
svc=SVC(kernel='linear',degree=1,gamma='auto',C=1)
svc.fit(x_train,y_train)
print("The accuracy on training set(svc) under linear:{:3f}".format(svc.score(x_train, y_train)))
print("The accuracy on training set(svc) under linear:{:3f}".format(svc.score(x_test,y_test)))

#高斯核函数
svc=SVC(kernel='rbf',degree=1,gamma='auto',C=1)
svc.fit(x_train,y_train)
print("The accuracy on training set(svc) under rbf(gauass):{:3f}".format(svc.score(x_train, y_train)))
print("The accuracy on training set(set) under rbf(gauass):{:3f}".format(svc.score(x_train, y_train)))

#多层感知机
mlp=MLPClassifier(random_state=0,max_iter=1000,alpha=1)
mlp.fit(x_train,y_train)
print("Training set score(mlp):{:3f}".format(mlp.score(x_train,y_train)))
print("Test set score(mlp):{:3f}".format(mlp.score(x_test,y_test)))

#决策树
tree=DecisionTreeClassifier(random_state=0,max_depth=4)
tree.fit(x_train, y_train)
print("Accuracy on training set(tree):{:3f}".format(tree.score(x_train,y_train)))
print("Accuracy on training set(tree):{:3f}".format(tree.score(x_test,y_test)))
