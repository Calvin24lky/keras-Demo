import numpy as np
np.random.seed(1337) #

from keras.models import Sequential # Sequential指按顺序添加网络层
from keras.layers import Dense # Dense指全连接层
import matplotlib.pyplot as plt

# create some data
X = np.linspace(-1, 1, 200)
np.random.shuffle(X) # shuffle 洗牌，原本x是-1到1步进，现在洗乱X

Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (X.shape[0],)) # 加点噪声

# plot data
plt.scatter(X, Y) # scatter：散点图
plt.show()

# 160个训练数据，40个测试数据
X_train, Y_train = X[:160], Y[:160]
X_test, Y_test = X[160:], Y[160:]

# build a neural network by keras
model = Sequential()
model.add(Dense(input_dim=1, output_dim=1)) # 第一层需定义输入维度，后面的层以上一层输出为默认输入维度

# choose loss function and optimizing method
model.compile(loss='mse', optimizer='sgd')

# 开始训练
print('Trainning -----------')

for step in range(301):
    loss = model.train_on_batch(X_train,Y_train)
    if step % 100 == 0:
        print('train loss: ', loss)

# 测试数据
print('\nTesting -----------')

loss = model.evaluate(X_test, Y_test, batch_size=40)
print('test loss: ', loss)

# 取出第1层的权重值
W, b = model.layers[0].get_weights()
print('Weights= ', W, '\nbiases= ', b)

# 画出预测曲线
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()