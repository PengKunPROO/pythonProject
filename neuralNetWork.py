import math

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)  # 1代表从第一堆种子里取随机数，这样想取相同的种子的话就可以每次从1这个堆中取出了

X,Y=load_planar_dataset()  # 获取数据 X是包含特征的numpy数组 Y是包含标签的numpy数组，即红：0，蓝：1

# region 画图，直观看出数据
# # 此图不是线性可分类的，所以用逻辑回归的方式进行分类效果不好
# plt.scatter(X[0, :], X[1, :], c=Y.reshape(X[0, :].shape), s=40, cmap=plt.cm.Spectral)
# # plt.show()
# x_shape=X.shape
# y_shape=Y.shape
# number=X.shape[1]  # 这个数据集应该是每一列是一个数据，因为一个数据有两个特征
# print(x_shape,y_shape,'there are '+str(number)+' samples')
# endregion

# -------------------------START------------------------- #


def layer_size(x_para, y_para):
    n_x = x_para.shape[0]  # 这是输入层大小
    n_h = 4  # 这是隐藏层大小，因为只有一个隐藏层所以不加n_h1这种标记了
    n_y = y_para.shape[0]  # 这是输出层大小，因为Y代表的是所有样本的特征，就是答案的意思，所以Y的行数就是要输出的大小
    return n_x, n_h, n_y


def initialize_parameters(n_x,n_h,n_y):  # 主要是初始化w1，b1，一个通式就是，有多少隐藏层，就得初始化多少个W和B
    """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        params -- 在字典中存储以下信息:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x)*0.01  # 因为是W*X，所以W1一定是（4，2）4代表第一层的大小，2是代表每一个特征（或者说前一层的神经元个数）都需要一个参数w
    b1 = np.zeros(n_h, 1)
    w2 = np.random.randn(n_y, n_h)*0.01  # 乘以0.01是为了加快收敛速度，而w2的维度是因为前一层的神经元数目是当前w的列数，当前层的神经元数是当前w的行数
    b2 = np.zeros(n_y, 1)

    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {'w1': w1,
                  'b1': b1,
                  'w2': w2,
                  'b2': b2}
    return parameters  # 最后返回一个字典，可以通过字典得到需要的参数值并进行更新


def sigmoid_tanh(para):
    """
    :param para: 参数是传入的值
    :return: 需要返回一个由tanh计算过的值
    """
    temp_value = np.tanh(para)
    return temp_value


def forward_propragation(x_input, parameters):
    """
    Argument:
    :param x_input: 用于输入样本数据，其shape是（n_x，m）nx是输入层节点数目，m是总共样本的数量
    :param parameters: 用于将w，b参数写入网络中并进行计算，得到输出
    :return:
    """

    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    Z1 = np.dot(w1, x_input)+b1
    A1 = sigmoid_tanh(Z1)
    Z2 = np.dot(w2, A1)+b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, x_input.shape[1]))  # 保证A2即输出层的shape为（1，400）

    dictionary={'A1': A1,
                'Z1': Z1,
                'A2': A2,
                'Z2': Z2
                }

    return A2, dictionary  # 返回A2是因为需要方便的得到A2，而用dictionary需要











