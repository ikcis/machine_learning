import numpy as np
import matplotlib.pyplot as plt


def func(x):
    return 4.8 * x + 5.2


num = 10
std = 0.25

x = np.linspace(0, 1, num)
x = np.expand_dims(x, 0)
y = func(x) + np.random.normal(scale=std, size=x.shape)

print(x.shape)
print(y.shape)

a = np.random.normal(size=(3,5))
print(a)