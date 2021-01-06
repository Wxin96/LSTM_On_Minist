import torch
import torchvision
import torch.nn as nn
import torch.utils.data as Data

# hyper parameters
# 迭代次数
EPOCH = 2
# 样本训练个数
BATCH_SIZE = 64
# 学习率
LR = 0.01
# 输入节点数
INPUT_SIZE = 28
# 输出节点数
OUTPUT_SIZE = 28

torch_data = torchvision.datasets.MNIST(
    root='./',  # 数据集位置
    train=True,  # 如果为True则为训练集，如果为False则为测试集
    transform=torchvision.transforms.ToTensor(),  # 将图片转化成取值[0,1]的Tensor用于网络处理
    download=True
)

train_loader = Data.DataLoader(dataset=torch_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./',
                                       train=False,
                                       transform=torchvision.transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor) / 255.0
test_y = test_data.test_labels.squeeze()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
print(rnn)

optimize = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        b_x = x.view(-1, 28, 28)
        b_y = y
        output = rnn(b_x)
        loss = loss_func(output, b_y)
        optimize.zero_grad()
        loss.backward()
        optimize.step()

        if step % 50 == 0:
            test_output = rnn(test_x)
            # test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
            # accuracy = float((pred_y == test_y).astype(int).sum()) / float(test_y.size())
            # accuracy  = sum(pred_y==test_y)/float(test_y.size())
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

test_output = rnn(test_x[:20].view(-1, 28, 28))
pred_y = torch.max(test_output, 1)[1].data.squeeze()
print(pred_y.data.numpy(), 'prediction number')
print(test_y[:20], 'real number')
