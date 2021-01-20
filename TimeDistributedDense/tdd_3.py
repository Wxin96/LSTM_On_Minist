from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
from keras.utils import plot_model

# 多对多（有TimeDistributed）

# prepare sequence
length = 5
seq = array([i / float(length) for i in range(length)])

# input (1, 5, 1) (batch size, time step, x's dimension)
X = seq.reshape(1, length, 1)
# output (1, 5, 1)
y = seq.reshape(1, length, 1)

# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 1000
# create LSTM
model = Sequential()
# 第一层 LSTM,
# units = 5 (units: 正整数，输出空间的维度, 单个LSTM的输出维度)
# return_sequences = true, 输出每个h_t
model.add(LSTM(n_neurons, batch_input_shape=(1, length, 1), return_sequences=True))
model.add(TimeDistributed(Dense(units=1, batch_input_shape=(1, 5)), batch_input_shape=(1, 5, 5)))

model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0, :, 0]:
    print('%.1f' % value)

plot_model(model, to_file='model_3.png', show_shapes=True)
