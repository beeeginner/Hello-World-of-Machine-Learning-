# -*- coding: GBK -*-
#导入需要的库
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold




#读取csv文件内容,header=0说明第一行是列名称

data = pd.read_csv('credit-overdue.csv',header=0)

data_array=np.array(data)

#读取自变量和因变量
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values.reshape(-1,1)

#添加常数项
X = np.insert(X,0,1,axis = 1)

#初始化参数
theta = np.zeros((X.shape[1],1))
alpha = 0.01
epsilon = 1e-5 #收敛阈值

#sigmoid函数
def sigmoid(z):
    return 1/(1+np.exp(-z))

#定义损失函数
def cost_function(X,y,theta):
    m = len(y)
    h = sigmoid(np.dot(X,theta))
    #交叉熵损失函数,比极大似然的相反数收敛速率更快
    J = (-1/m)*np.sum(y*np.log(h)+(1-y)*np.log(1-h))
    return J
    
#定义梯度下降函数    
def gradiant_descent(X,y,theta,alpha,epsilon):
    """
    X: 输入特征矩阵，每行代表一个样本，每列代表一个特征
    y: 输出标签，为0或1
    alpha: 学习率
    theta: 系数向量
    epsilon: 收敛阈值
    """
    m = len(y)
    cost = cost_function(X,y,theta)
    cost_old = cost + 5*epsilon #将cost_old初始化,5是我随便选的为了让他开始
    num_iters = 0
    #条件收敛停止梯度下降
    while abs(cost_old - cost)>epsilon:
        cost_old = cost
        h = sigmoid(np.dot(X,theta))
        gradiant = np.dot(X.T,(h-y))/m
        theta = theta - alpha*gradiant
        cost = cost_function(X,y,theta)
        num_iters+=1
        if num_iters % 100 == 0:
            print(f'iteration {num_iters}, Cost function value: {cost}')
    return theta
      
    
#k折交叉验证
k = 5
skf = StratifiedKFold(n_splits=k)
accuracies = []
confusion = np.zeros((2,2))
for train_index,test_index in skf.split(X,y):
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    theta = np.zeros((X.shape[1],1))
    theta = gradiant_descent(X_train,y_train,theta,alpha,epsilon)
    #大于0.5的认为是1，小于0.5的认为是0
    y_pred = np.round(sigmoid(np.dot(X_test,theta)))
    #confusion_matrix不接受binary值所以不做round来一个预测值y_pred0
    y_pred0 = sigmoid(np.dot(X_test,theta))
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)

#输出参数
print('Theta: ',theta)
print('KFOLD accuracies(5 times):',accuracies)

