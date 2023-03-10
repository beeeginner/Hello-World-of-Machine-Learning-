import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scikitplot as skplt
import matplotlib.pyplot as plt


#sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
#训练出来的结果
theta = np.array([0.05492737,2.24140229,-2.37434832])

#读取数据并且划分一个测试集出来
data = pd.read_csv('credit-overdue.csv',header=0)
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values.reshape(-1,1)
X = np.insert(X,0,1,axis=1)
_,X_test,_,y_test = train_test_split(X,y,test_size=0.5,random_state =0)


#模型训练的值
y_pred = np.round(sigmoid(np.dot(X_test,theta)))
fig = plt.figure(figsize=(15,6))
ax1 = fig.add_subplot(121)
skplt.metrics.plot_confusion_matrix(y_test,y_pred,title="Confusion Matrix",cmap="Oranges",ax=ax1)

plt.show()

