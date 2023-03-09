# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#theta是模型的计算结果我这里直接抄下来了，需要自己算的话麻烦去计算核心的py文件再算一次，theta0是常数项。
theta = np.array([0.05492737,2.24140229,-2.37434832])

df=pd.read_csv('credit-overdue.csv')
p=df['debt']
q=df['income']
plt.rcParams['font.sans-serif'] = 'SimHei' #用于正常显示中文
plt.title('数据散点图及其划分曲线') # 添加标题
plt.xlabel('debt')# 添加x轴的名称
plt.ylabel('income')# 添加y轴的名称
map_size={0:20,1:100}
size=list(map(lambda x:map_size[x],df['overdue']))
plt.scatter(p,q,s=size,c=df['overdue'],marker='v')
k1=theta[1]
k2=theta[2]
b1=theta[0]
px1=np.arange(0,2.5,0.01)
px2=-b1/k2-k1/k2*px1
plt.plot(px1,px2)
plt.show()