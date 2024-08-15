import pandas as pd
import sklearn
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from torchvision import transforms
from torch.nn import Linear, Sequential, ReLU, Softmax
from torch.utils.data import TensorDataset, DataLoader \

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'

EPOCH = 80
BATCH_SIZE = 6
LR = 0.02

# 读取数据集
iris = datasets.load_iris()
data = iris.data
target = iris.target

# 对数据打乱
np.random.seed(116)
indices = np.arange(data.shape[0])  # 生成长度为data长度的随机序列
np.random.shuffle(indices)  # 对序列进行随机打乱
data = data[indices]  # 用刚才生成的随机序列，把data和target再进行打乱
target = target[indices]

# 转换为tensor(张量), .float()转换为float32，因为Linear()默认为float32
data = torch.from_numpy(data).float()  # 150 * 4
# target需要保证为long类型
target = torch.from_numpy(target).long()

train_data = data[0:120]
test_data = data[120:]
train_target = target[0:120]
test_target = target[120:]

# 展示数据
flowerName = ['Setosa', 'Versicolour', 'Virginica']
featureName = ['sepal length', 'sepal width', 'petal length', 'petal width']
print(data[0:5])
print(target[0:5])

# 加载数据集
dataset_train = TensorDataset(train_data, train_target)
dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataset_test = TensorDataset(test_data, test_target)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)


# 创建感知器模型
class irisClassify(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = Sequential(
            Linear(4, 10),
            ReLU(),
            Linear(10, 3),
            Softmax(dim=1)  # 在下标为1的维度，因为有两个维度（batch_size,num_class),num_class就是类别，这里是将类别转换为概率。
        )

    def forward(self, data):
        output = self.module(data)
        return output

# 定义分类器、损失函数、梯度优化器，在定义优化器时还定义了学习率
iris_recognition_module = irisClassify()
loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(iris_recognition_module.parameters(), lr=LR)

# 测试模型
input = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
input.shape
output = iris_recognition_module(input)
output.dtype

iterator = 0
bestRightNum = 0
for epoch in range(EPOCH):
    # 训练模型
    _ = iris_recognition_module.train();
    print(f"------第{epoch}次训练------")
    for data in dataloader_train:  # 这里的遍历，每次提供的是一批，而不是一个样本。单位是一个batch
        trainData, trainTarget = data  # dataloader当初定义的时候就是有着两个张量，所以这里直接就挨个赋值
        print("train_data:", trainData)
        output = iris_recognition_module(trainData)
        print("output:", output)
        loss_out = loss(output, trainTarget)
        optim.zero_grad()  # 梯度清零
        loss_out.backward()  #  反向传播计算参数
        optim.step()  # 参数更新
        if iterator % 10 == 0:
            print(f"第{iterator}次的损失为: {loss_out:.6f}")
        iterator += 1
    # 测试模型
    _ = iris_recognition_module.eval();
    with torch.no_grad():
        total_right = 0
        for data in dataloader_test:
            testData, testTraget = data  # (6, 4), (6,)
            output = iris_recognition_module(testData)
            right = (output.argmax(1) == testTraget).sum()
            total_right += right
        print(f"第{epoch}次训练的准确率为:  {1.0 * total_right / len(test_data):.3f}")
        if total_right >= bestRightNum or epoch == EPOCH - 1:
            bestRightNum = total_right
            path = f"./models/iris_reiris_recognition_model_{epoch}.pth"
            torch.save(iris_recognition_module.state_dict(), path)

# 手动测试模型是否正确

# 4.6,3.2,1.4,0.2,Iris-setosa
# 5.3,3.7,1.5,0.2,Iris-setosa
# 5.0,3.3,1.4,0.2,Iris-setosa
# 7.0,3.2,4.7,1.4,Iris-versicolor
# 6.4,3.2,4.5,1.5,Iris-versicolor

# 5.1,2.5,3.0,1.1,Iris-versicolor
# 5.7,2.8,4.1,1.3,Iris-versicolor
# 6.3,3.3,6.0,2.5,Iris-virginica
# 5.8,2.7,5.1,1.9,Iris-virginica
# 7.1,3.0,5.9,2.1,Iris-virginica

with torch.no_grad():
    testData1 = torch.tensor([[4.6, 3.2, 1.4, 0.2],
                              [5.3, 3.7, 1.5, 0.2],
                              [5.0, 3.3, 1.4, 0.2],
                              [7.0, 3.2, 4.7, 1.4],
                              [6.4, 3.2, 4.5, 1.5]], dtype=torch.float32)
    testTarget1 = [0, 0, 0, 1, 1]
    predict1 = iris_recognition_module(testData1)
    predict1 = predict1.argmax(1)
    predict1
    testData2 = torch.tensor([[5.1, 2.5, 3.0, 1.1],
                              [5.7, 2.8, 4.1, 1.3],
                              [6.3, 3.3, 6.0, 2.5],
                              [5.8, 2.7, 5.1, 1.9],
                              [7.1, 3.0, 5.9, 2.1]], dtype=torch.float32)
    testTarget2 = [1, 1, 2, 2, 2]
    predict2 = iris_recognition_module(testData2)
    predict2 = predict2.argmax(1)  # argmax会返回张量中第二维最大值的索引。因为分类器的output实际上是三个类别的概率，所以返回三者中的最大值就是类别
    predict2
    testData3 = torch.tensor([[10, 10, 100, 10],
                              [20, 20, 200, 20],
                              [5.1, 2.5, 3.0, 1.1],
                              [78, 46, 90, 1],
                              [1200, 3400, 8900, 1200]], dtype=torch.float32)
    # 这4项数据都是凭空捏造的，应该均不属于这三类
    predict3 = iris_recognition_module(testData3)
    predict3 = predict3.argmax(1)
    predict3
