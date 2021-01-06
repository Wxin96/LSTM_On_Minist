import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data

# hyper parameters
EPOCH = 2
BATCH_SIZE = 64
INPUT_SIZE = 28
OUTPUT_SIZE = 28
LR = 0.01

torch_data = torchvision.datasets.MNIST(
    root='MINIST',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_loader = Data.DataLoader(dataset=torch_data,batch_size=BATCH_SIZE,shuffle=True)

# #test the output dimision
# test_data = torchvision.datasets.MNIST(root='MINIST',train=False)
# #这时候经过transom后的结果任然是1-255，变成了tensor的类型
# print(test_data.test_data[10])
# #这个时候的结果变成了【0-1】的范围
# # test_x = test_data.test_data.type(torch.FloatTensor)[10]/255.
# test_x = test_data.test_data[10]/255.
# print(test_x)
# test_y = test_data.test_labels
# print(test_y)



test_data = torchvision.datasets.MNIST(root='MINIST',
                                       train=False,
                                       transform=torchvision.transforms.ToTensor())
test_x = test_data.test_data.type(torch.FloatTensor)/255.0
test_y = test_data.test_labels.squeeze()


class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Linear(64,10)

    def forward(self, x):
        r_out ,(h_n , h_c) = self.rnn(x,None)
        out = self.out(r_out[:,-1,:])
        return out

rnn = RNN()
print(rnn)

optimize = torch.optim.Adam(rnn.parameters(),lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step ,(x,y) in enumerate(train_loader):
        b_x= x.view(-1,28,28)
        b_y = y
        output = rnn(b_x)
        loss = loss_func(output,b_y)
        optimize.zero_grad()
        loss.backward()
        optimize.step()

        if step%50 ==0:
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