import time
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary
from KANConv import KAN_Convolutional_Layer
from torch import nn
from kan import KAN


class MyDataset:
    def __init__(self):
        super().__init__()
        # 打开存储数据的txt文件，数据处理后将结果以元组方式保存在train_data里
        f1 = open('./Xm_all_2.txt')
        data = np.loadtxt(f1, delimiter=',')
        # data = data.reshape((10, 10, 20, 250, 4)).transpose(0, 1, 3, 2, 4)
        self.data = data.reshape((10, 10, 20, 250, 4))[:, :, :, :, :2].transpose(0, 1, 3, 2, 4)
        f2 = open('./X.txt')
        label = np.loadtxt(f2, delimiter=',')
        # label = label.reshape((10, 250, 6))[:, :, [0, 3, 1, 4]]
        self.label = label.reshape((10, 250, 6))[:, :, [0, 3]]
        f1_ = open('./Xm_test_2.txt')
        data_ = np.loadtxt(f1_, delimiter=',')
        # data_ = data.reshape((10, 10, 20, 250, 4)).transpose(0, 1, 3, 2, 4)
        self.data_ = data_.reshape((10, 10, 20, 250, 4))[:, :, :, :, :2].transpose(0, 1, 3, 2, 4)
        f2_ = open('./X_test.txt')
        label_ = np.loadtxt(f2_, delimiter=',')
        # label_ = label_.reshape((10, 250, 6))[:, :, [0, 3, 1, 4]]
        self.label_ = label_.reshape((10, 250, 6))[:, :, [0, 3]]
        self.c = 2

    def train_set(self):
        train_data = []
        train_label = []
        train_factor_set = np.ones((100, 230, 2))
        for i in range(10):  # 条
            for j in range(10):  # 重复次数
                for k in range(230):
                    x_t = copy.deepcopy(self.data[i, j, k:k + 10])
                    # factor = np.array([x_t[0, :, 0], x_t[0, :, 1]]).transpose((1, 0))
                    factor = np.array([x_t[0, 0, 0], x_t[0, 0, 1]])
                    # factor = np.array([0, 0])
                    train_factor_set[i, k] = factor
                    x_t = (x_t - factor).transpose(2, 0, 1)  # (10,20,2)
                    # y_t = copy.deepcopy(self.label[i, k + 10])
                    y_t = copy.deepcopy(self.label[i, k + 9])
                    y_t = y_t - factor
                    # y_t = np.repeat(y_t, 20, axis=0).reshape((self.c, 20))
                    train_data.append(((x_t / 85) .tolist()))
                    train_label.append(((y_t / 85).tolist()))
        train_factor_set.tofile("train_factor_set_100_230_2.dat", sep=",", format="%f")
        return TensorDataset(torch.tensor(train_data), torch.tensor(train_label))

    def test_set(self):
        test_data = []
        test_label = []
        n = 4
        n_ = 3
        test_factor_set = np.ones((n_, 230, 2))
        for i in range(n, n + n_):
            for j in range(0, 1):
                for k in range(230):
                    x_t = copy.deepcopy(self.data_[i, j, k:k + 10])
                    # factor = np.array([x_t[0, :, 0], x_t[0, :, 1]]).transpose((1, 0))
                    factor = np.array([x_t[0, 0, 0], x_t[0, 0, 1]])
                    # factor = np.array([0, 0])
                    test_factor_set[i-n, k] = factor
                    x_t =  (x_t - factor).transpose(2, 0, 1)   # (10,20,2)
                    # y_t = copy.deepcopy(self.label_[i, k + 10])
                    y_t = copy.deepcopy(self.label_[i, k + 9])
                    y_t = y_t - factor
                    # y_t = np.repeat(y_t, 20, axis=0).reshape((self.c, 20))
                    test_data.append(((x_t / 85).tolist()))
                    test_label.append(((y_t / 85).tolist()))
        test_factor_set.tofile("test_factor_set_10_230_2.dat", sep=",", format="%f")
        return TensorDataset(torch.tensor(test_data), torch.tensor(test_label))


class KKAN_Convolutional_Network(nn.Module):
    def __init__(self, device: str = 'cuda'):
        super().__init__()
        self.conv1 = KAN_Convolutional_Layer(
            n_convs=8,
            kernel_size=(2, 3, 3),
            padding=(1, 1),
            device=device
        )

        self.conv2 = KAN_Convolutional_Layer(
            n_convs=16,
            kernel_size=(8, 3, 3),
            padding=(1, 0),
            device=device
        )

        self.pool1 = nn.AvgPool2d(
            kernel_size=(2, 2)
        )

        self.flat = nn.Flatten()
        self.kan = KAN(width=[128, 16, 2], symbolic_enabled=False, base_fun=nn.Tanh(), device='cuda:0')  # 神经元个数

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.flat(x)
        x = self.kan(x)
        return x


def adjust_learning_rate(optimizer):
    # 自定义学习率下降规则（相当于每过opt.epochs轮, 学习率就会乘上0.1）
    # 其中opt.lr为初始学习率, opt.epochs为自定义值
    lr = optimizer.param_groups[0]['lr'] * 10
    return lr


def model_fit():
    # 实例化对象
    M = MyDataset()
    train_data = M.train_set()
    test_data = M.test_set()

    # 将数据集导入DataLoader，进行shuffle以及选取batch_size
    train_dataloader = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_data, batch_size=256, shuffle=True, num_workers=0)
    # Windows里num_works只能为0，其他值会报错
    if torch.cuda.is_available():
        device = torch.device("cuda")
        de_str = 'cuda'
        print("使用GPU训练中：{}".format(torch.cuda.get_device_name()))
    else:
        device = torch.device("cpu")
        de_str = 'cpu'
        print("使用CPU训练")
    mymodule = KKAN_Convolutional_Network(device=de_str)
    mymodule = mymodule.to(device)  # 模型转移GPU
    loss = torch.nn.MSELoss()
    learn_step = 0.001
    optim = torch.optim.Adam(mymodule.parameters(), lr=learn_step)
    epoch = 2000
    train_loss = []
    test_loss = []

    for i in range(epoch):
        mymodule.train()  # 模型在训练状态
        epoch_loss = 0
        train_step = 0

        # if (i+1) % 1 == 0:
        #     for p in optim.param_groups:
        #         p['lr'] *= 10
        # lr_list.append(optim.state_dict()['param_groups'][0]['lr'])

        # lr = adjust_learning_rate(optim)
        # for param_group in optim.param_groups:
        #     param_group["lr"] = lr

        # print('---------learning rate', optim.param_groups[0]["lr"])

        for data in train_dataloader:
            encoder_data, targets = data
            encoder_data = encoder_data.to(device)
            targets = targets.to(device)
            outputs = mymodule(encoder_data)
            result_loss = loss(outputs.to(torch.float32), targets.to(torch.float32))

            optim.zero_grad()
            result_loss.backward()
            optim.step()

            train_step += 1
            epoch_loss += result_loss.item()
            # if train_step % 100 == 0:
            #     print("第{}轮的第{}次训练的loss:{}".format((i + 1), train_step, result_loss.item()))
        train_loss.append(epoch_loss / train_step)
        print("第{}轮训练的loss:{}".format((i + 1), epoch_loss / train_step))
        # 在测试集上面的效果
        mymodule.eval()  # 在验证状态
        val_loss = 0
        tt = 0
        with torch.no_grad():  # 验证的部分，不是训练所以不要带入梯度
            for test_data in test_dataloader:
                tt += 1
                encoder_data, label = test_data
                encoder_data = encoder_data.to(device)
                label = label.to(device)

                outputs_ = mymodule(encoder_data)
                test_result_loss = loss(outputs_, label)
                val_loss += test_result_loss.item()
            test_loss.append(val_loss / tt)
            print("第{}轮训练在测试集上的Loss为{}".format((i + 1), test_result_loss.item()))

        if (i + 1) % 100 == 0:
            # 保存模型
            torch.save(mymodule.state_dict(), "train_model_1/mymodule_{}.pth".format((i + 1)))
    # torch.save(mymodule.state_dict(), "mymodule_{}.pth".format((epoch + 1)))
    loss_value = np.array(train_loss)
    loss_value.tofile("train_loss/loss.dat", sep=",", format="%f")
    val_loss_value = np.array(test_loss)
    val_loss_value.tofile("train_loss/val_loss.dat", sep=",", format="%f")
    k = [i - 0.5 for i in range(len(loss_value))]
    # # 绘制训练 & 验证的损失值
    plt.figure(1)
    plt.plot(k, loss_value)
    plt.plot(k, val_loss_value)
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('Epoch')
    plt.legend('Train Loss', loc='upper right')
    plt.show()
    return


def model_predict():
    # 实例化对象

    test_data = MyDataset().test_set()
    n = 1
    factor_set = np.fromfile("test_factor_set_10_230_2.dat", dtype=float, sep=",").reshape((n * 230, 2))

    # 将数据集导入DataLoader，进行shuffle以及选取batch_size
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    # Windows里num_works只能为0，其他值会报错
    if torch.cuda.is_available():
        device = torch.device("cuda")
        de_str = "cuda"
        print("使用GPU训练中：{}".format(torch.cuda.get_device_name()))
    else:
        device = torch.device("cpu")
        de_str = "cpu"
        print("使用CPU训练")
    mymodule = KKAN_Convolutional_Network(de_str)
    mymodule = mymodule.to(device)  # 模型转移GPU
    mymodule.load_state_dict(torch.load("train_model_1/mymodule_{}.pth".format(1900)))
    # mymodule.load_state_dict(torch.load("train_model_1/mymodule_prue.pth"))
    # for name, param in mymodule.named_parameters():
    #     print(name, '-->', param.type(), '-->', param.dtype, '-->', param.shape)
    # print('*************grad**********', mymodule.EncoderLayer_0.Norm1[1].running_mean.data, '@@@@@@@')
    # print('*************grad**********', mymodule.EncoderLayer_0.Norm1[1].running_var.data, '@@@@@@@')
    # print(mymodule)
    mymodule.eval()

    out_all = []
    label = []
    encoder_input = []
    t1 = time.time()
    with torch.no_grad():
        for data in test_dataloader:
            encoder_data, targets = data
            encoder_data = encoder_data.to(device)
            outputs = mymodule(encoder_data)
            out_all.append(outputs[0].tolist())
            label.append(targets[0].tolist())
            encoder_input.append(encoder_data[0].tolist())
        # mymodule.kan.prune()
        # torch.save(mymodule.state_dict(), "train_model_1/mymodule_prue.pth")
        # mymodule.kan.plot(mask=True)
        # return

    out_all = np.array(out_all)
    label = np.array(label)
    # out_mean = np.mean(out_all, 2) * 85 + factor_set
    # label_mean = np.mean(label, 2) * 85 + factor_set
    out_mean = out_all * 85 + factor_set
    label_mean = label * 85 + factor_set
    print((time.time() - t1) / 230)
    det = out_mean - label_mean
    error = np.sqrt(np.mean(np.power(det[:, 0], 2) + np.power(det[:, 1], 2)))
    print(error)
    out_mean.tofile("kan_3.dat", sep=",", format="%f")
    plt.figure(1)
    plt.plot(out_mean[:, 0], out_mean[:, 1], label='estimation')
    plt.plot(label_mean[:, 0], label_mean[:, 1], linestyle='--', label='label')

    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend(loc='upper right')

    plt.show()
    return


if __name__ == '__main__':
    # model_fit()
    model_predict()
    # train_ = MyTrainDataset()
    # train_.plot_data()
