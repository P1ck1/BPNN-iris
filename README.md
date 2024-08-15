# BPNN原理部分：

![1723534297005](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723534297005.png)

## 基本概念

BPNN，反向神经传播网络，Back Propagation Neural Network。

分为输入层，隐含层，输出层。网络的入参是一个n维向量，出参是一个m维向量。对于入参，出参，均可以有多种理解方式。

###### 入参

可以是一张图片的像素灰度集合，一个xi代表一个像素点。出参：

###### 出参

对于手写数字识别，可以有十个输出，分别代表图片是0~9的概率。

对于人脸识别任务，可以输出特定长度的特征向量，代表某个人的人脸特征。

## 正向传播过程

从最左边开始，对于一个神经元来说，所有的输入与其权重w进行乘积求和，再加上偏置b，然后再放入激活函数中，成为该神经元的输出，即作为下一层的输入。

## 反向传播，训练过程

***这里重点理解这一部分***

##### 代价函数

首先要知道代价函数（误差函数）就是网络给出的预测值与目标值的误差，这个目标是是随着数据集给出的。

有均方误差，交叉熵误差等，此处用均方误差理解。

![1723535305214](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723535305214.png)

由于代价函数就是一个**由各层神经元各权重和各偏移量构成的多元函数**，训练神经网络的目的在于找到一组**使得该代价函数取得最小值**的各层神经元权重矩阵和偏移向量。这里要引入梯度下降法。**梯度就是由多元函数各自变量的偏导数组成的向量，一个自变量的偏导数反应了该自变量对函数值增长的影响度，所以一个多元函数的梯度指向的就是在多维空间中该函数值增长最快的方向，而反方向就是函数值下降最快的方向，沿此方向移动自变量**，即所谓梯度下降。

**所以其实求梯度，就是求偏导数的过程，这个是核心，要想知道某一参数如何调整，就是求他的梯度，实际上也就是求该参数对于误差函数的偏导数**



这里涉及到几个变量。

![1723535537960](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723535537960.png)

输入x，权重w，偏置b，z是w*x(o)的和+b，o是经过激活函数激活的输出

![1723535629896](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723535629896.png)

对于某一样本来说，梯度如下

![1723535816604](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723535816604.png)

这里的每一个元素都是一个矩阵或者向量

首先考虑权重w的偏导

![1723535927813](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723535927813.png)

这个推导的意义是，将对w的偏导，转换为，该层神经元的敏感程度***δ****上一层的输出。

这个***δ***被称为第i ii层神经元的灵敏度，反映了第i ii层神经元对网络总输出误差的影响程度。

对偏置求导道理相同，且较为简单

![1723536115367](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723536115367.png)

**接下来，问题转换为求*δ***，即对z的偏导，这个z指的是激活函数的入参，即各输入加权求和再加上偏置。

链式求导![1723536253384](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723536253384.png)

此处要注意了，L关于o1求偏导，o1**会作为下一层所有神经元的输入**，所以此处是**多元函数**求导

![1723536422541](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723536422541.png)



这里的f′(z1(1))指的是激活函数对于z求导。

这里就体现了**误差从输出层反向传播到输入层**，因为前一层的δ，可以从后一层的δ通过矩阵运算得到。

同理：

![1723536499064](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723536499064.png)

最后需要收束，因为最后的输出o3直接就是误差函数的自变量了，不会成为新的输入了。

![1723536655769](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723536655769.png)

要理解这个，误差函数长这样：![1723536722055](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723536722055.png)

然后，这个y^就是o(3),所以L对o(3)求偏导就是很简单的形式（就是x的n次方的求导法则）,之后就是激活函数对z求偏导。

![1723537096088](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723537096088.png)

## 总结

以上篇幅较为繁琐，主要目的就是理解：

1.网络的传播都是通过矩阵运算实现的，详细的讲解了矩阵是如何生成的，以及背后的原理

2.要让参数调整，就是使得误差下降，往哪里下降？求梯度。如何求？就是链式求导，求偏导数

3.这里要进行补充

​	（1）存在学习率，就是说，求得梯度之后，不是立马就减去梯度，比如参数是x，实际上是x-=αk。α是学习率，k是梯度

​	（2）参数如何初始化，即一开始的参数要设为什么。可以均匀分布，也可以高斯分布

# 代码学习：

**这里学习了两种代码，均是使用BPNN进行鸢尾花(iris)分类：**

**1.没有使用pytorch的框架的，而是基于原理的朴实复刻，复刻了向前传播、反向传播、求梯度、更新参数等全流程操作。**

**2.第二种是使用pytorch框架的bpnn算法，实现鸢尾花(iris)分类**

详细学习笔记均在代码注释中，这里做简略总结

## 朴实复刻：

手动实现了BPNN的几个步骤。一共只有三层，输入层，隐含层，输出层

### 初始化参数

使用正态分布随机取值，范围是-0.01~0.01存储在字典中。

### 向前传播

首先从字典中取值，然后使用numpy的矩阵运算与数学运算手动向前传播

### 代价函数

使用交叉熵，使用numpy函数进行计算

### 反向传播

计算dw1、db1、dw2、db2，之后再更新w和b。这里和之前学习的一样，从右往左反向计算，梯度=灵敏度*前一层的输出

### 更新参数

使用之前计算的dw1、db1、dw2、db2，再结合之前设定好的学习率，进行更新

### 构造神经网络

首先设定好输入节点数，隐藏层节点数，输出层节点数

然后初始化参数

然后迭代num_iterations次，每次迭代过程中依次执行之前的函数：前向传播、计算代价函数、反向传播、更新参数

### 模型评估

使用训练好的参数，对测试集的样本进行预测得到结果output，然后遍历查看是否与目标值target相同，得到正确率。

### 主函数

在主函数中，首先获取数据集，这里使用的是sklearn的在线导入数据集。然后随机将数据集划分为训练集和测试集，随后进行了独热编码的转换。

独热编码，只有一位是1，其他地方是0.作用是使得不同类别之间距离相等，适用于分类数据。这里是将标签target（共有0 1 2)三种转换为独热编码，比如可以转换为001 010 100.

训练完毕之后进行模型评估

![1723706553738](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723706553738.png )

## 使用pytorch框架

对于这个场景，使用pytorch有几点不同：

1.数据需要进一步使用torch.from_numpy转换为tensor张量，是pytorch的要求

2.加载数据要使用dataloder，在这里还要指定batch_size

下面开始讲解代码：

```python
class irisClassify(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = Sequential(
            Linear(4, 10),
            ReLU(),
            Linear(10, 3),
            Softmax(dim=1)
        )

​    def forward(self, data):
​        output = self.module(data)
​        return output
```

这里就是定义了神经网络，很简洁，这就是基于pytorch框架进行定义神经网络模型，现在说明几点：

1.我们定义的神经网络继承自 `torch.nn.Module`，这是所有神经网络模块的基类。

2.首先进行初始化，且在初始化的第一步就是调用父类的构造函数

3.然后self.module = Sequential( ... ) 创建了一个 torch.nn.Sequential 容器，它将按照顺序包含多个层。Sequential 是一个有序的容器，用于包装一系列层，以便它们可以按顺序被调用。

这里就是需要我们理解以及进行的工作了：

首先Linear(4, 10) 是一个全连接层（也称为线性层），它将输入特征从4维映射到10维。这里的4对应于鸢尾花数据集的四个特征：萼片长度、萼片宽度、花瓣长度和花瓣宽度。

ReLU() 是一个激活函数层。

另一个 Linear(10, 3) 层将10维的输出再次映射到3维，这里的3对应于鸢尾花数据集中的三个类别。

Softmax(dim=1)` 是一个softmax层，用于将前一层的输出转换为概率分布。`dim=1` 指定了应用softmax函数的维度，这里指的是第二个维度，即特征维度。

这里涉及到一个问题，就是数据的一系列传导问题，一直困扰着我，在最后讲解完全部代码进行一下梳理。

**训练过程：**

```python
for epoch in range(EPOCH):

# 训练模型

​    _ = iris_recognition_module.train();
​    print(f"------第{epoch}次训练------")
​    for data in dataloader_train:  # 这里的遍历，每次提供的是一批，而不是一个样本。单位是一个batch
​        trainData, trainTarget = data  # dataloader当初定义的时候就是有着两个张量，所以这里直接就挨个赋值
​        output = iris_recognition_module(trainData)
​        loss_out = loss(output, trainTarget)
​        optim.zero_grad()  # 梯度清零
​        loss_out.backward()  #  反向传播计算参数
​        optim.step()  # 参数更新
​        if iterator % 10 == 0:
​            print(f"第{iterator}次的损失为: {loss_out:.6f}")
​        iterator += 1

# 测试模型

​    _ = iris_recognition_module.eval();
​    with torch.no_grad():
​        total_right = 0
​        for data in dataloader_test:
​            testData, testTraget = data  # (6, 4), (6,)
​            output = iris_recognition_module(testData)
​            right = (output.argmax(1) == testTraget).sum()
​            total_right += right
​        print(f"第{epoch}次训练的准确率为:  {1.0 * total_right / len(test_data):.3f}")
​        if total_right >= bestRightNum or epoch == EPOCH - 1:
​            bestRightNum = total_right
​            path = f"./models/iris_reiris_recognition_model_{epoch}.pth"
​            torch.save(iris_recognition_module.state_dict(), path)


```

在每一世代中，以batch为单位，训练完所有的样本。这里又体现了pytorch框架的作用：损失函数在之前定义好的，也无需自己编写。然后关于梯度计算、参数更新，也只需要调用pytorch自带的优化器的对应函数。每训练十次打印一次准确率。

训练完当前世代的数据之后，进行当前世代的准确率评估。也是将测试集数据进行预测，看准确率。**且这里也涉及到一个pytorch神经网络应用的常规流程**：保存准确率高的参数。这里维护了一个最高准确率，如果当前世代的准确率高于历史最好，就会更新最高准确率，并进行保存。

### 结果：

![1723709638199](C:\Users\22602\AppData\Roaming\Typora\typora-user-images\1723709638199.png)

### 难点

最后，数据合乎逻辑的传递，这个问题我认为和理解pytorch的全流程也有很大关系。

首先，data的大小是（120，4）。然后，在实际训练过程中，以batch为单位进行训练，train_data变成（6，4)

然后进入网络，首先是Linear(4, 10)。(6,10)*(4,10)=(6,10),然后激活，然后再经过Linear(10, 3)。

(6,10)\*(10,3)=(6,3)然后再Softmax(dim=1)，这里就可以理解这个dim=1的作用了，是对于第二维度进行转换为概率。

这里我打印出来看一下，助于理解

train_data: tensor([[7.2000, 3.6000, 6.1000, 2.5000],
        [6.3000, 2.5000, 4.9000, 1.5000],
        [5.2000, 2.7000, 3.9000, 1.4000],
        [7.1000, 3.0000, 5.9000, 2.1000],
        [6.1000, 3.0000, 4.9000, 1.8000],
        [4.9000, 2.4000, 3.3000, 1.0000]])
output: tensor([[0.1952, 0.2035, 0.6014],
        [0.2268, 0.1865, 0.5867],
        [0.2457, 0.2266, 0.5277],
        [0.2019, 0.1853, 0.6127],
        [0.2216, 0.2125, 0.5659],
        [0.2661, 0.2160, 0.5179]], grad_fn=<SoftmaxBackward0>)

