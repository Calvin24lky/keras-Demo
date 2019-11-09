import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import SimpleRNN, Activation, Dense
from keras.optimizers import Adam

TIME_STEPS = 28   # 时间长度，每次读取1行mnist图像的值，共28行，即28次读取
INPUT_SIZE = 28   # 一次读入的数据是28个pixels
BATCH_SIZE = 50   # 每批训练50张图片
BATCH_INDEX = 0   #
OUTPUT_SIZE = 10  # 输入one-hot
CELL_SIZE = 50    # 相当于hidden layer的单元数
LR = 0.001

# 下载数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据预处理
X_train = X_train.reshape(-1, 28, 28) / 255.      # normalize
X_test = X_test.reshape(-1, 28, 28) / 255.        # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 建立RNN网络
model = Sequential()

# RNN cell
model.add(SimpleRNN(
    batch_input_shape=(None, TIME_STEPS, INPUT_SIZE), # or input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    unroll=True,
))

# output layer
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# 优化器与损失函数
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 开始训练
print('Trainning -----------')

# for step in range(6001):
#     X_batch = X_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :, :]
#     y_batch = y_train[BATCH_INDEX: BATCH_INDEX + BATCH_SIZE, :]
#     cost = model.train_on_batch(X_batch, y_batch)
#     BATCH_INDEX += BATCH_SIZE
#     BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX
#
#     # if step % 500 == 0:
#     #     cost, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=True)
#     #     print('test cost: ', cost, 'test accuracy: ', accuracy)

model.fit(X_train, y_train, epochs=1, batch_size=50,)

loss, accuracy = model.evaluate(X_test, y_test, batch_size=y_test.shape[0], verbose=True)
print('test cost: ', loss)
print('test accuracy: ', accuracy)