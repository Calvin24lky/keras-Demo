from keras.datasets import mnist
from keras.layers import Dense, LSTM, Activation
from keras.utils import to_categorical
from keras.models import Sequential

# parameters for LSTM
CELL = 30  # 神经元个数
TIME_STEPS = 28  # 时间序列长度
INPUT_SIZE = 28 # 输入序列

# data preprocessing: tofloat32, normalization, one_hot encoding
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# build model
model = Sequential()
model.add(LSTM(units=CELL, input_shape=(TIME_STEPS, INPUT_SIZE)))
model.add(Dense(10))
model.add(Activation('softmax'))

# loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train
model.fit(x_train, y_train, epochs=5, batch_size=128, verbose=True)

model.summary()

# test
print('\nTesting -----------')
loss, accuracy = model.evaluate(x_test, y_test, verbose=True)

print('test loss: ', loss)
print('test acc: ', accuracy)
