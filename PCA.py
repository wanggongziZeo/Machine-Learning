# -*- coding:utf-8 -*-
# 使用PCA对数据集进行降维

#-----------------------------------------------------------------------------------------------#
# 导入包
import numpy as np 
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold
#-----------------------------------------------------------------------------------------------#


#-----------------------------------------------------------------------------------------------#
# 加载数据集
def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target
#-----------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------#
# PCA测试函数
def test_PCA(*data):
    X, y = data
    pca = decomposition.PCA(n_components=None)
    pca.fit(X, y)
    print("Explained Variance Ratio:\n %s" % str(pca.explained_variance_ratio_))

X, y = load_data()
# test_PCA(X, y)

#-----------------------------------------------------------------------------------------------#
# 绘制将维后的样本分布图
# 由上知，降维后的数据维度为2
def plot_PCA(*data):
    X, y = data
    pca = decomposition.PCA(n_components = 2)
    pca.fit(X)
    X_r = pca.transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5),
                (0.4,0.6,0), (0.6,0.4,0), (0,0.6,0.4), (0.5,0.3,0.2))
    for label, color in zip(np.unique(y),colors):
        position=y==label
        ax.scatter(X_r[position,0],X_r[position,1],label = "target = %d"%label,color = color)
    
    ax.set_xlabel("x[0]")
    ax.set_ylabel("y[0]")
    ax.legend(loc = 'best')
    ax.set_title("PCA")
    plt.show()

# plot_PCA(X, y)
#-----------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------#
# 绘制将维后的样本分布图
# 由上知，降维后的数据维度为2
def plot_IncrementalPCA(*data):
    X, y = data
    pca = decomposition.IncrementalPCA(n_components = 2,batch_size = 10)
    pca.partial_fit(X)
    print("n_samples_seen_:", pca.n_samples_seen_)
    X_r = pca.transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5),
                (0.4,0.6,0), (0.6,0.4,0), (0,0.6,0.4), (0.5,0.3,0.2))
    for label, color in zip(np.unique(y),colors):
        position=y==label
        ax.scatter(X_r[position,0],X_r[position,1],label = "target = %d"%label,color = color)
    
    ax.set_xlabel("x[0]")
    ax.set_ylabel("y[0]")
    ax.legend(loc = 'best')
    ax.set_title("PCA")
    plt.show()

# plot_IncrementalPCA(X, y)
#-----------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------#
# 使用KernelPCA降维的测试函数
def test_KernelPCA(*data):
    X, y = data
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for kernel in kernels:
        kpca = decomposition.KernelPCA(n_components = None, kernel = kernel)
        kpca.fit(X)
        print("kernel = %s --> lambdas: %s\n" % (kernel, len(kpca.lambdas_)))
        print()

# test_KernelPCA(X, y)
#-----------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------#
# 绘制降维之后的数据
def plot_KernelPCA(*data):
    X , y = data
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    fig = plt.figure()
    colors = ((1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5),
                (0.4,0.6,0), (0.6,0.4,0), (0,0.6,0.4), (0.5,0.3,0.2))
    for i, kernel in enumerate(kernels):
        kpca = decomposition.KernelPCA(n_components = 2, kernel = kernel)
        kpca.fit(X)
        X_r = kpca.transform(X)
        ax = fig.add_subplot(2,2,i+1)
        for label, color in zip(np.unique(y), colors):
            position=y==label
            ax.scatter(X_r[position, 0], X_r[position, 1], label = "target = %d" % label, color = color)
        ax.set_xlabel("X[0]")
        ax.set_ylabel("X[1]")
        ax.legend(loc = 'best')
    plt.suptitle("KPCA")
    plt.show()

# plot_KernelPCA(X, y)
#-----------------------------------------------------------------------------------------------#
    
#-----------------------------------------------------------------------------------------------#
# 考察多项式和函数的参数的影响
def test_KPCA_poly(*data):
    X, y = data
    fig = plt.figure()
    colors = ((1,0,0), (0,1,0), (0,0,1), (0.5,0.5,0), (0,0.5,0.5), (0.5,0,0.5),
                (0.4,0.6,0), (0.6,0.4,0), (0,0.6,0.4), (0.5,0.3,0.2))
    Params = [(3,1,1), (3,10,1),(3,1,10),(3,10,10),(10,1,1),(10,10,1),(10,1,10),(10,10,10)]

    for i,(p, gamma, r) in enumerate(Params):
        kpca = decomposition.KernelPCA(n_components = 2, kernel = 'poly', gamma = gamma, degree = p, coef0 = r)
        kpca.fit(X)
        X_r = kpca.transform(X)
        ax = fig.add_subplot(2, 4, i+1)
        for label, color in zip(np.unique(y), colors):
            position=y==label
            ax.scatter(X_r[position, 0], X_r[position, 1], label = "Target = %d" % label, color = color)

        ax.set_xlabel("X[0]")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("X[1]")
        ax.legend(loc = 'best')
        ax.set_title(r"$(%s (x \cdot z+1)+%s)^{%s}$" % (gamma, r, p))
    plt.suptitle("KPCA-poly")
    plt.show()

test_KPCA_poly(X, y)

































