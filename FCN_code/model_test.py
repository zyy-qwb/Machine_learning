import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import keras
from keras.models import load_model
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 导入数据集
dataset = pd.read_csv("data.csv")
# print(dataset)

# 将数据进行归一化
sc = MinMaxScaler(feature_range=(0, 1))
scaled = sc.fit_transform(dataset)
# print(scaled)

# 将归一化好的数据转化为datafrme格式，方便后续处理
dataset_sc = pd.DataFrame(scaled)
# print(dataset_sc)

# 将数据集中的特征和标签找出来
X = dataset_sc.iloc[:, :-1]
Y = dataset_sc.iloc[:, -1]

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=42)

# 加载已经训练好的模型
model = load_model("model.h5")

# 利用训练好的模型进行预测
yhat = model.predict(x_test)
# print("归一化之前的值", yhat)


# 进行预测值的反归一化
inv_yhat = concatenate((x_test, yhat), axis=1)
inv_yhat = sc.inverse_transform(inv_yhat)
# print(inv_yhat)
prediction = inv_yhat[:, 6]
# print("归一化之后的值", prediction)


# 将y_test维度的转化
y_test = np.array(y_test)
# print(y_test)
y_test = np.reshape(y_test, (y_test.shape[0], 1))
# print(y_test)


# 反向缩放真实值
inv_y = concatenate((x_test, y_test), axis=1)
inv_y = sc.inverse_transform(inv_y)
real = inv_y[:, 6]
# print(real)


# 计算rmse和MAPE
rmse = sqrt(mean_squared_error(real, prediction))
mape = np.mean(np.abs((real- prediction)/prediction))

# 打印rmse和mape
print('rsme', rmse)
print('mape', mape)


# 画出真实值和预测值的对比值

plt.plot(prediction, label='预测值')
plt.plot(real, label="真实值")
plt.title("全连接神经网络空气质量预测对比图")
plt.legend()
plt.show()