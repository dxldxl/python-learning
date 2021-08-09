# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:34:04 2021

@author: cui
"""

import numpy as np
import pandas as pd

data=np.array([
    [1, 1, 1, 1, 1, 1, 0.697, 0.460, 1],
    [2, 1, 2, 1, 1, 1, 0.774, 0.376, 1],
    [2, 1, 1, 1, 1, 1, 0.634, 0.264, 1],
    [1, 1, 2, 1, 1, 1, 0.608, 0.318, 1],
    [3, 1, 1, 1, 1, 1, 0.556, 0.215, 1],
    [1, 2, 1, 1, 2, 2, 0.403, 0.237, 1],
    [2, 2, 1, 2, 2, 2, 0.481, 0.149, 1],
    [2, 2, 1, 1, 2, 1, 0.437, 0.211, 1],
    [2, 2, 2, 2, 2, 1, 0.666, 0.091, 0],
    [1, 3, 3, 1, 3, 2, 0.243, 0.267, 0],
    [3, 3, 3, 3, 3, 1, 0.245, 0.057, 0],
    [3, 1, 1, 3, 3, 2, 0.343, 0.099, 0],
    [1, 2, 1, 2, 1, 1, 0.639, 0.161, 0],
    [3, 2, 2, 2, 1, 1, 0.657, 0.198, 0],
    [2, 2, 1, 1, 2, 2, 0.360, 0.370, 0],
    [3, 1, 1, 3, 3, 1, 0.593, 0.042, 0],
    [1, 1, 2, 2, 2, 1, 0.719, 0.103, 0]
    ])

column=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率', '好坏']
dataSet=pd.DataFrame(data,columns=column)

class Relief(object):
    def __init__(self,data):
        self.data=data
        #样本
        self.x=data.iloc[:,:-1]
        #标记
        self.y=data.iloc[:,-1]
        #样本个数
        self.m=len(data)
        #属性个数
        self.n=len(self.x.iloc[0])
        
    #计算两个样本之间的距离
    def calc_distance(self,sample1,sample2):
        distance=0
        for i in range(self.n):
            #前6个为离散属性
            if i<=5:
                if sample1[i]==sample2[i]:
                    distance +=0
                else:
                   distance +=1
            #对离散属性
            else:
                #2范数
                distance +=(sample1[i]-sample2[i])**2
        return distance
    
    #计算两个样本之间的diff值
    def calc_diff(self,sample1,sample2,attr_index):
        #对于离散属性
        if attr_index<=5:
            if sample1[attr_index]==sample2[attr_index]:
                return 0
            else:
                return 1
        #对连续属性
        else:
            return abs(sample1[attr_index]-sample2[attr_index])
            
        
    def find_near_hit(self,sample_index):
        #对第i个样本的标记
        label=self.y[sample_index]
        hit_samples=self.data[self.data['好坏']==label]
        #筛选出属性同类样本的属性部分
        hit_samples=hit_samples.iloc[:,:-1]
        #设定一个很大的数来初始化最近距离
        nearest_distance=float('inf')
        #初始化xi_nh
        xi_nh=0
        for i in range(len(hit_samples)):#遍历所有样本
            if i==sample_index:
                pass
            else:
                distance=self.calc_distance(self.x.iloc[sample_index],hit_samples.iloc[i])
                if distance<nearest_distance:
                    nearest_distance=distance
                    #用于计算的xi_nh
                    xi_nh=hit_samples.iloc[i]
        return xi_nh
    
    def find_near_miss(self,sample_index):
        label=self.y[sample_index]  #该样本的标记
        miss_samples=self.data[self.data['好坏']!=label]
        #筛选出不同类样本的属性部分
        miss_samples=miss_samples.iloc[:,:-1]
        #设定一个很大的数来初始化最近距离
        nearest_distance=float('inf')
        xi_nm=0  #初始化xi_nm
        for i in range(len(miss_samples)):
            distance=self.calc_distance(self.x.iloc[sample_index],miss_samples.iloc[i])
            if distance< nearest_distance:
                nearest_distance=distance
                xi_nm=miss_samples.iloc[i]
        return xi_nm
    
    def fit(self):
        result=[]
        #遍历所有属性
        for attr_index in range(self.n):
            delta=0
            #遍历所有样本
            for sample_index in range(self.m):
                #猜中近邻
                xi_nh=self.find_near_hit(sample_index)
                #猜错近邻
                xi_nm=self.find_near_miss(sample_index)
                diff_nh=self.calc_diff(self.x.iloc[sample_index],xi_nh,attr_index)
                diff_nm=self.calc_diff(self.x.iloc[sample_index],xi_nm,attr_index)
                delta +=-diff_nh**2+diff_nm**2
            result.append(delta)
        return result
    
relief=Relief(dataSet)
result=relief.fit()
sort_result=[]
for i in range(len(result)):
    sort_result.append((column[i],result[i]))
sort_result=sorted(sort_result,key=lambda x:x[1],reverse=True)
print(sort_result)

        
        
        