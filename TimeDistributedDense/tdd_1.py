from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# 5个样本，1个时间步长，1个特征

# prepare sequence
length = 5
seq = array([i/float(length) for i in range(length)])
X = seq.reshape(len(seq), 1, 1)
y = seq.reshape(len(seq), 1)

# define LSTM configuration
n_neurons = 5
n_batch = 5
n_epoch = 1000

# create LSTM
model = Sequential()
# 第一层 LSTM
model.add(LSTM(n_neurons, input_shape=(1, 1)))
# 第二层 Dense
model.add(Dense(1))
# 损失函数、优化器
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result:
	print('%.1f' % value)