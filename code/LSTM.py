import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

np.random.seed(0)


def calculate_mae(y_true, y_pred):
    # 平均绝对误差
    mae = np.mean(np.abs(y_true - y_pred))
    return mae


true_data = pd.read_csv('31_1135_1352.csv')  # 填你自己的数据地址

target = '总流量'

# 这里加一些数据的预处理, 最后需要的格式是pd.series

true_data = np.array(true_data['总流量'])


# 训练集和测试集的尺寸划分
test_size = 0.15
train_size = 0.85
# 标准化处理
scaler_train = MinMaxScaler(feature_range=(0, 1))
scaler_test = MinMaxScaler(feature_range=(0, 1))
train_data = true_data[:int(train_size * len(true_data))]
test_data = true_data[-int(test_size * len(true_data)):]
print("训练集尺寸:", len(train_data))
print("测试集尺寸:", len(test_data))
train_data_normalized = scaler_train.fit_transform(train_data.reshape(-1, 1))
test_data_normalized = scaler_test.fit_transform(test_data.reshape(-1, 1))
# 转化为深度学习模型需要的类型Tensor
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)
test_data_normalized = torch.FloatTensor(test_data_normalized).view(-1)


def create_inout_sequences(input_data, tw, pre_len):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        if (i + tw + 9) > len(input_data):
            break
        train_label = input_data[i + tw:i + tw + pre_len]
        inout_seq.append((train_seq, train_label))
    return inout_seq

pre_len = 9
train_window = 30

# 定义训练器的的输入
train_inout_seq = create_inout_sequences(train_data_normalized, train_window, pre_len)
if int(input("train or not")):
    Train = 1
else:
    Train = 0
class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=350, output_dim=1):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(1)

        h0_lstm = torch.zeros(1, self.hidden_dim).to(x.device)
        c0_lstm = torch.zeros(1, self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0_lstm, c0_lstm))
        out = out[:, -1]
        out = self.fc(out)

        return out


lstm_model = LSTM(input_dim=1, output_dim=pre_len, hidden_dim=train_window)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
epochs = 20


if Train:
    losss = []
    lstm_model.train()  # 训练模式
    start_time = time.time()  # 计算起始时间
    for i in range(epochs):
        for seq, labels in train_inout_seq:
            lstm_model.train()
            optimizer.zero_grad()

            y_pred = lstm_model(seq)

            single_loss = loss_function(y_pred, labels)

            single_loss.backward()
            optimizer.step()
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
            losss.append(single_loss.detach().numpy())
    torch.save(lstm_model.state_dict(), 'save_model.pth')
    print(f"模型已保存,用时:{(time.time() - start_time) / 60:.4f} min")
    plt.plot(losss)
    # 设置图表标题和坐标轴标签
    plt.title('Training Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    # 保存图表到本地
    plt.savefig('training_error.png')
else:
    # 加载模型进行预测
    lstm_model.load_state_dict(torch.load('save_model.pth'))
    lstm_model.eval()  # 评估模式
    results = []
    reals = []
    losss = []
    test_inout_seq = create_inout_sequences(test_data_normalized, train_window, pre_len)
    for seq, labels in train_inout_seq:
        pred = lstm_model(seq)[0].item()
        results.append(pred)
        mae = calculate_mae(pred, labels.detach().numpy())  # MAE误差计算绝对值(预测值  - 真实值)
        reals.append(labels.detach().numpy())
        losss.append(mae)
    # 创建折线图
    plt.plot(reals, label='real', color='blue')  # 实际值
    plt.plot(results, label='forecast', color='red', linestyle='--')  # 预测值

    # 增强视觉效果
    plt.grid(True)
    plt.title('real vs forecast')
    plt.xlabel('time')
    plt.ylabel('value')
    plt.legend()
    plt.savefig('test——results.png')
    results = scaler_test.inverse_transform(test_data_normalized.view(-1, 1))
    results = results.flatten()
    print(len(results),len(losss))
    new_df = pd.DataFrame({
        '预测结果': results
    })
    new_df.to_csv('测试结果.csv', index =False, encoding='utf-8')
    new_df = pd.DataFrame({
        'mae': losss
    })
    new_df.to_csv('测试误差.csv', index=False, encoding='utf-8')
