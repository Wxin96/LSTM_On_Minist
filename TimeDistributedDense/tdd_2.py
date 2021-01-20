from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import plot_model
# 多对一（没有TimeDistributed）

# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])

# input (1, 5, 1) (batch size, time step, x's dimension)
X = seq.reshape(1, length, 1)
# output (1, 5)
y = seq.reshape(1, length)

# define LSTM configuration
n_neurons = 5
n_batch = 1
n_epoch = 500

# create LSTM
model = Sequential()
# 第一层 LSTM,
# units = 5 (units: 正整数，输出空间的维度, 单个LSTM的输出维度)
# return_sequences = false, 只输出最后的h_t
model.add(LSTM(units=n_neurons, batch_input_shape=(1, length, 1)))

model.add(Dense(units=5, batch_input_shape=(1, 5)))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:]:
	print('%.1f' % value)


plot_model(model, to_file='model_2.png', show_shapes=True)






