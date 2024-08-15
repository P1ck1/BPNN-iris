import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets

'''
    构建一个具有1个隐藏层的神经网络，隐层的大小为10
'''

# 1.初始化参数
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    # 权重和偏置矩阵
    w1 = np.random.randn(n_h, n_x) * 0.01  # 取值是从正态分布取，矩阵大小是10*4，每列代表一个输入对应的权重
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    # 存储参数
    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters


# 2.前向传播
def forward_propagation(X, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    # 通过前向传播来计算a2。且对于输入来说是全批量处理，因为X一次就包含了120个样本的数据
    z1 = np.dot(w1, X) + b1     # 这个地方需注意矩阵加法：虽然(w1*X)和b1的维度不同，但可以相加
    a1 = np.tanh(z1)            # 使用tanh作为第一层的激活函数（双曲正切）
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))  # 使用sigmoid作为第二层的激活函数

    # 通过字典存储参数
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}

    return a2, cache


# 3.计算代价函数
def compute_cost(a2, Y):
    m = Y.shape[1]      #总的样本数

    # 采用交叉熵作为代价函数。a2经过激活函数处理，所以在0-1
    logprobs = np.multiply(np.log(a2), Y) + np.multiply((1 - Y), np.log(1 - a2))
    cost = - np.sum(logprobs) / m

    return cost


# 4.反向传播（计算代价函数的导数）
def backward_propagation(parameters, cache, X, Y):
    m = Y.shape[1]

    w2 = parameters['w2']
    # a是激活后的
    a1 = cache['a1']
    a2 = cache['a2']

    # 反向传播，计算dw1、db1、dw2、db2，之后再更新w和b。这里和之前学习的一样，从右往左反向计算，梯度=灵敏度*前一层的输出
    dz2 = a2 - Y  # 这是简化，实际上应该是损失函数对a2求导。这里是简化
    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, X.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}

    return grads


# 5.更新参数
def update_parameters(parameters, grads, learning_rate=0.4):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    # 更新参数，考虑学习率
    w1 = w1 - dw1 * learning_rate
    b1 = b1 - db1 * learning_rate
    w2 = w2 - dw2 * learning_rate
    b2 = b2 - db2 * learning_rate

    parameters = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    return parameters


# 建立神经网络
def bpnn(X, Y, n_h, n_input, n_output, num_iterations=5000, print_cost=False):
    np.random.seed(3)

    n_x = n_input           # 输入层节点数
    n_y = n_output          # 输出层节点数

    # 1.初始化参数
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 梯度下降循环
    for i in range(0, num_iterations):
        # 2.前向传播
        a2, cache = forward_propagation(X, parameters)
        # 3.计算代价函数
        cost = compute_cost(a2, Y)
        # 4.反向传播
        grads = backward_propagation(parameters, cache, X, Y)
        # 5.更新参数
        parameters = update_parameters(parameters, grads)

        # 每10次迭代，输出一次代价函数
        if print_cost and i % 100 == 0:
            print('迭代第%i次，代价函数为：%f' % (i, cost))

    return parameters


# 6.模型评估
def judge(parameters, x_test, y_test):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, x_test) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = 1 / (1 + np.exp(-z2))

    # 结果的维度
    n_rows = a2.shape[0]
    n_cols = a2.shape[1]

    # 预测值结果存储
    output = np.empty(shape=(n_rows, n_cols), dtype=int)

    for i in range(n_rows):
        for j in range(n_cols):
            if a2[i][j] > 0.5:
                output[i][j] = 1
            else:
                output[i][j] = 0

    # 将one-hot编码反转为标签
    output = encoder.inverse_transform(output.T)
    output = output.reshape(1, output.shape[0])
    output = output.flatten()

    print('预测结果：', output)
    print('真实结果：', y_test)

    count = 0
    for k in range(0, n_cols):
        if output[k] == y_test[k]:
            count = count + 1
        else:
            print('错误分类样本序号：', k + 1)

    accuracy = count / int(a2.shape[1]) * 100
    print('BP神经网络准确率：%.2f%%' % accuracy)
    return output


#随机抽取80%的训练集和20%的测试集
def divideData():
    completeData = np.c_[iris.data, iris.target.T]
    np.random.shuffle(completeData)
    trainData = completeData[range(int(length * 0.8)), :]
    testData = completeData[range(int(length * 0.8), length), :]
    return [trainData, testData]
# range()返回的是数字序列，所以range(int(length * 0.8))是0——119的序列，range(int(length * 0.8), length)是120-149的序列
# 所以，range是给出了行的范围，而这个':'是给出了列的范围，代表选取所有列

if __name__ == "__main__":

    # 获取iris数据集
    iris = datasets.load_iris()

    # 随机获取80%训练集和20%测试集
    length = len(iris.target)
    [trainData, testData] = divideData()
    X=trainData[:,0:4].T  # 所有行，0-3列
    Y=trainData[:,-1].T  # 最后一列
    # 转置后，X变为（4，120）4是特征数量，120是样本数量，即（特征数量，样本数量）

    # 将标签转换为one-hot编码,便于计算
    encoder = OneHotEncoder()  # 独热编码，只有一位是1，其他地方是0.作用是使得不同类别之间距离相等
    Y = encoder.fit_transform(Y.reshape(Y.shape[0], 1))
    # 这里的Y.shape[0]指的是y的第一个维度的大小，由于Y是一维的，所以就是Y的大小
    # 然后Y.shape是重塑，将Y重塑为(Y.shape[0], 1)大小的二维数组，即Y.shape[0]行，每行只有一个元素
    Y = Y.toarray().T  # 稀疏转密集，在这里实际上没有变换。然后转置，这样每列代表一个标签的独热编码（一个列向量代表一个）
    Y = Y.astype('uint8')

    # 输入4个节点，隐层10个节点，输出3个节点，迭代5000次
    parameters = bpnn(X, Y, n_h=10, n_input=4, n_output=3, num_iterations=5000, print_cost=True)

    x_test = testData[:,0:4].T
    y_test = testData[:,-1].T

    result = judge(parameters, x_test, y_test)
