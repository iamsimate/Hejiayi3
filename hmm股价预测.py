# -*- coding: utf-8 -*-
"""
Created on Thu May 19 16:57:53 2022

@author: Administrator
"""

import datetime
import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM

#数据处理
df = pd.read_excel("stto.xlsx", header=0)#600511.SH
print("原始数据的大小：", df.shape)
print("原始数据的列名", df.columns)
df['时间'] = pd.to_datetime(df['时间'])
df.reset_index(inplace=True, drop=False)
df.drop(['index','交易日期', '开盘价', '最高价','换手率', '最低价', 'pe', 'pb'], axis=1,inplace=True)
print(df.head())
dates = df['时间'][1:]
close_v = df['收盘价']#预测目标
volume = df['成交量'][1:]
diff = np.diff(close_v)#前一天和后一天收盘价的差
#获得输入数据
X = np.column_stack([diff, volume])
print("输入数据的大小：", X.shape) 

#异常值的处理
min = X.mean(axis=0)[0] - 8*X.std(axis=0)[0]   #最小值
max = X.mean(axis=0)[0] + 8*X.std(axis=0)[0]  #最大值
X = pd.DataFrame(X)
#异常值设为均值
for i in range(len(X)):  #dataframe的遍历
    if (X.loc[i, 0]< min) | (X.loc[i, 0] > max):
            X.loc[i, 0] = X.mean(axis=0)[0]
            
#数据集的划分
X_Test = X.iloc[:-70]#倒数30index前的作训练集
X_Pre = X.iloc[-70:]#倒数30index作测试集
print("训练集的大小：", X_Test.shape)     
#print("测试集的大小：", X_Pre.shape)      

#模型的搭建

model = GaussianHMM(n_components=5, covariance_type='diag', n_iter=1000,min_covar=.1) 
model.fit(X_Test)

print("隐藏状态的个数", model.n_components)  
print("均值矩阵")
print(model.means_)
print("协方差矩阵")
print(model.covars_)
print("状态转移矩阵--A")
print(model.transmat_)

#训练数据的隐藏状态划分
expected_returns_volumes = np.dot(model.transmat_, model.means_)
expected_returns = expected_returns_volumes[:,0]        
predicted_price = []  #预测值
current_price = close_v.iloc[-70]

for i in range(len(X_Pre)):
   hidden_states = model.predict(X_Pre.iloc[i].values.reshape(1,2))  #将预测的第一组作为初始值
   predicted_price.append(current_price+expected_returns[hidden_states])
   current_price = predicted_price[i]

x = dates[-69:]
y_act = close_v[-69:]
y_pre = pd.Series(predicted_price[:-1])
plt.figure(figsize=(8,6))
plt.plot_date(x, y_act,linestyle="-",marker="None",color='g')
plt.plot_date(x, y_pre,linestyle="-",marker="None",color='r')
plt.legend(['Actual', 'Predicted'])
plt.show()