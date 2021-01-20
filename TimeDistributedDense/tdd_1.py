from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model

# 5个样本，1个时间步长，1个特征
# 一对一

# prepare sequence
length = 5
seq = array([i / float(length) for i in range(length)])
# input (5, 1, 1) (batch size, time step, x's dimension)
X = seq.reshape(len(seq), 1, 1)
y = seq.reshape(len(seq), 1)

# define LSTM configuration
n_neurons = 5
n_batch = 5
n_epoch = 1000

# create LSTM
model = Sequential()
# 第一层 LSTM,
# units = 5 (units: 正整数，输出空间的维度, 单个LSTM的输出维度)
model.add(LSTM(units=n_neurons, batch_input_shape=(5, 1, 1)))
# 第二层 Dense, （在第一层之后，你就不再需要指定输入的尺寸了，可省略）
model.add(Dense(units=1, batch_input_shape=(5, 5)))
# 损失函数、优化器
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result:
    print('%.1f' % value)

plot_model(model, to_file='model_1.png', show_shapes=True, expand_nested=True)
