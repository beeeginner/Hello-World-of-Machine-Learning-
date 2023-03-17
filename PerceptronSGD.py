import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold


# 导入数据集,并且分离出自变量因变量且在特征中加入常数列
data = pd.read_csv('credit-overdue.csv',header=0)
data_array = np.array(data)

labels = data.iloc[:,-1].values
features = data.iloc[:,:-1].values
features = np.insert(features,0,1,axis=1)
#数据预处理，在这个例子中，我们的正例为0，反例为1，这个不适合perception，我们需要把正例换成1，反例换成-1！
labels = np.where(labels==0,1,-1)

class PerceptronSGD:
    def __init__(self,lr=0.01,epoches=100):
        self.lr = lr
        self.epoches = epoches
        self.weights = None
    def fit(self,X,y):
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.epoches):
            for i in range(X.shape[0]):
                if y[i]*np.dot(X[i],self.weights)<=0:
                    #分类错误的时候,梯度下降更新W
                    self.weights +=self.lr*y[i]*X[i]
                    
        return self.weights
        
    def predict(self,X):
        return np.sign(np.dot(X,self.weights))

#交叉验证KFoLD
k = 3
skf = StratifiedKFold(n_splits=k)
accuracies = []
W = []

for train_index,test_index in skf.split(features,labels):
    X_train,X_test = features[train_index],features[test_index]
    y_train,y_test = labels[train_index],labels[test_index]
    classifier = PerceptronSGD()
    classifier.fit(X_train,y_train)
    y_pred = classifier.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    accuracies.append(accuracy)
    W0 = classifier.weights
    W.append(W0)
    
#输出参数
print(W)
print('KFOLD accuracies(3 times):',accuracies)