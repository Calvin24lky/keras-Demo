import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
from keras.optimizers import Adam

BATCH_START = 0
TIME_STEPS = 20   # 时间长度，每次读取20个值
INPUT_SIZE = 1   # 一次读入的数据是1个x
OUTPUT_SIZE = 1  # 输出一个y
BATCH_SIZE = 50   # 每批训练50张图片
CELL_SIZE = 20    # 相当于hidden layer的单元数
LR = 0.006

def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


# 建立LSTM-RNN网络
model = Sequential()

# LSTM-RNN cell
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),  # or input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    return_sequences=True,  # 默认为false 只在最后一个t时刻输出y，改为true之后，每个t时刻都输出y
    stateful=True,  # 前面读28次读完一张图，下一个28次与前面的图是无关的，但是这里预测曲线是有关的，所以改为True
))

# output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))

# 优化器与损失函数
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='mse',)

# 开始训练
print('Trainning -----------')

for step in range(501):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch, Y_batch, xs = get_batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)
    plt.plot(xs[0, :], Y_batch[0].flatten(), 'r', xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.1)
    if step % 10 == 0:
        print('train cost: ', cost)

plot_model(model, to_file='lstm_regression.png')