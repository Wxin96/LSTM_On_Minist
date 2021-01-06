import torchvision
import torch.utils.data as Data
import torch.nn as nn
import torch


def test1_train_data():
    # 样本训练个数
    BATCH_SIZE = 64

    torch_data = torchvision.datasets.MNIST(
        root='./',  # 数据集位置
        train=True,  # 如果为True则为训练集，如果为False则为测试集
        transform=torchvision.transforms.ToTensor(),  # 将图片转化成取值[0,1]的Tensor用于网络处理
        download=True
    )
    print(torch_data.train_data.shape)
    print(torch_data.train_labels.shape)
    train_loader = Data.DataLoader(dataset=torch_data, batch_size=BATCH_SIZE, shuffle=True)
    print(type(torch_data))
    print(type(train_loader))

    for step, (x, y) in enumerate(train_loader):
        b_x = x.view(-1, 28, 28)
        print(b_x.shape)
    #     print(step)
    #     print(x)
    #     print(y)


def test2_test_data():
    test_data = torchvision.datasets.MNIST(root='./',
                                           train=False,
                                           transform=torchvision.transforms.ToTensor(),
                                           download=True)
    print(test_data.test_data.shape)

    test_x = test_data.test_data.type(torch.FloatTensor) / 255.0
    test_y = test_data.test_labels.squeeze()


def test3_lstm():
    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    print(input.shape)
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    output, (hn, cn) = rnn(input, (h0, c0))


if __name__ == '__main__':
    # test1_train_data()
    # test2_test_data()
    test3_lstm()
