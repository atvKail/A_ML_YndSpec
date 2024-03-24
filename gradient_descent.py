import numpy as np


def LikeDerivative(func, x, dx=1e-9):
    return (func(x + dx) - func(x)) / dx


def condition(func, point, gradient, epsilon):
    return np.linalg.norm(func(point - gradient) - func(point)) >= epsilon


def gradient_descent(func, start_point, gamma, epsilon, steps):
    history = [[start_point]]

    point = start_point
    gradient = gamma * LikeDerivative(func, point)

    if steps == 0:
        while condition(func, point, gradient, epsilon):
            point -= gradient
            gradient = gamma * LikeDerivative(func, point)
            history.append([point])
    else:
        for _ in range(steps - 1):
            point -= gradient
            gradient = gamma * LikeDerivative(func, point)
            history.append([point])

    history.append([point - gradient])
    history = np.round(history, 3)
    history.reshape(-1, 1)
    return history


print(gradient_descent(lambda x:x**2, 2, 0.01, 0.0001, 0))
print(len(gradient_descent(lambda x:x**2, 2, 0.01, 0.0001, 0)))
