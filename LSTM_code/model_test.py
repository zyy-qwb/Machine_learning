# 导入库
import pandas as pd
import numpy as np
import matplotlib.pyplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import load_model
import matplotlib.pyplot as plt

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载历史数据文件
dataset = pd.read_csv('LBMA-GOLD.csv', index_col='Date')
# print(dataset)

# 设置训练集的长度
training_len = 1256 -200


# 获取测试集数据
test_set = dataset.iloc[training_len:, [0]]

# 将数据集进行归一化，方便神经网络的训练
sc = MinMaxScaler(feature_range=(0, 1))
test_set = sc.fit_transform(test_set)


# 设置放置测试数据特征和测试数据标签的列表
x_test = []
y_test = []



# 同理划分测试集数据
for i in range(5, len(test_set)):
    x_test.append(test_set[i - 5:i, 0])
    y_test.append(test_set[i, 0])


# 测试集变array并reshape为符合要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 5, 1))

# 导入模型
model = load_model('model.h5')


# 利用模型进行测试
predicted = model.predict(x_test)
# print(predicted.shape)

# 进行预测值的反归一化
prediction = sc.inverse_transform(predicted)
# print(prediction)

# 对测试集的标签进行反归一化

real = sc.inverse_transform(test_set[5:])
# print(real)


# 打印模型的评价指标
rmse = sqrt(mean_squared_error(prediction, real))
mape = np.mean(np.abs((real-prediction)/prediction))
print('rmse', rmse)
print('mape', mape)

# 绘制真实值和预测值的对比
plt.plot(real, label='真实值')
plt.plot(prediction, label='预测值')
plt.title("基于LSTM神经网络的黄金价格预测")
plt.legend()
plt.show()