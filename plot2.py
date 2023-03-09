import numpy as np
from matplotlib import pyplot as plt

# 梯度函数的导数
def gradJ1(theta):
    return 4 * theta


def gradJ2(theta):
    return 2 * theta


# 梯度函数
def f(x, y):
    return 2 * x ** 2 + y ** 2


def ff(x, y):
    return 2 * np.power(x, 2) + np.power(y, 2)


def train(theta1=2.241, theta2=-2.374,lr=0.05, epoch=50):
    t1 = [theta1]
    t2 = [theta2]
    for i in range(epoch):
        gradient = gradJ1(theta1)
        theta1 = theta1 - lr * gradient
        t1.append(theta1)
        gradient = gradJ2(theta2)
        theta2 = theta2 - lr * gradient
        t2.append(theta2)

    plt.figure(figsize=(10, 10))  # 设置画布大小
    x = np.linspace(-3, 3, 30)
    y = np.linspace(-3, 3, 30)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')  # 等高线图
    ax.scatter3D(t1, t2, ff(t1, t2), c='r', marker='o')
    # 调整观察角度和方位角。这里将俯仰角设为100度，把方位角调整为45度
    ax.view_init(45, 100)
    plt.show()
train()
