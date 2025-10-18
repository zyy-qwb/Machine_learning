import pandas as pd
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

data=pd.read_csv('train.csv')
data_test=pd.read_csv('test.csv')
# print(data.info())
# print(data_test.info())
#data.head()
#data.describe()
#data.dtypes

#删除非必要列
data=data.drop(['PassengerId','Name','Ticket'],axis=1)
data_test=data_test.drop(['Name','Ticket'],axis=1)
#处理性别字段
data['Gender']=data['Sex'].map({'female':0,'male':1}).astype(int)
data_test['Gender']=data_test['Sex'].map({'female':0,'male':1}).astype(int)
#data['Gender'].head()
#处理年龄字段 平均值，中位数
data['Age'].mean()
data['Age'].median()
data[data['Age']>60]
data[data['Age']>60][['Sex','Pclass','Survived']]
data[data['Age'].isnull()][['Sex','Pclass','Age']]

#绘制直方图
data['Age'].dropna().hist(bins=16,range=(0,80),alpha=0.5)
plt.show()