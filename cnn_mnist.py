from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils, plot_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam

# 下载数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# 数据预处理
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 建立cnn网络
model = Sequential()

# Conv layer 1 output shape (32, 28, 28)
model.add(Convolution2D(
    batch_input_shape=(None, 1, 28, 28),
    filters=32,
    kernel_size=5,
    strides=1,
    padding='same', # padding method
    data_format='channels_first',
    ))
model.add(Activation('relu'))

# Pooling layer 1 (max pooling) output shape (32, 14, 14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first',
    ))

# Conv layer 2 output shape (64, 14, 14)
model.add(Convolution2D(
    filters=64,
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first',
    ))
model.add(Activation('relu'))

# Pooling layer 2 (max pooling) output shape (64, 7, 7)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first',
    ))

# Fully connected layer 1 input shape (64*7*7 = 3136), output shape(1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
# 输出层
model.add(Dense(10))
model.add(Activation('softmax'))

# 优化器与损失函数
adam = Adam(lr=1e-4)
model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    )

# 开始训练
print('Trainning -----------')
model.fit(X_train, y_train, epochs=1, batch_size=64,)

# 测试数据
print('\nTesting -----------')
loss, accuracy = model.evaluate(X_test, y_test)

print('test loss: ', loss)
print('test acc: ', accuracy)

#plot_model(model, to_file='cnn_mnist.png')