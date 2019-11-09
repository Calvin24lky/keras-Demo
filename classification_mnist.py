import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# 下载mnist数据集
(Xraw_train, yraw_train), (Xraw_test, yraw_test) = mnist.load_data()

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据预处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255 # 原数据0-255， 标准化为0-1
X_test = X_test.reshape(X_test.shape[0], -1) / 255

y_train = np_utils.to_categorical(y_train, num_classes=10) #y由单值标签变为one-hot标签
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 建立网络
model = Sequential([
    Dense(32, input_dim=784),
    Activation('relu'),
    Dense(10), # input_dim默认为上一层的output_dim=32
    Activation('softmax') # 分类器
    ])

# 定义优化器
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# 定义损失函数,编译模型
model.compile(
    optimizer=rmsprop,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

# 开始训练
print('Trainning -----------')
model.fit(X_train, y_train, nb_epoch=2, batch_size=32)

# 测试数据
print('\nTesting -----------')
loss, accuracy = model.evaluate(X_test, y_test)
print('test loss: ', loss)
print('test acc: ', accuracy)

plot_model(model, to_file='model.png')



