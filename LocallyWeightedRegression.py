import matplotlib.pyplot as plt
import numpy as np
import math
import time

#def draw_graph():

if __name__ == "__main__":
    # Generate some data
    np.random.seed(0)
    M = 100
    X = np.random.rand(M, 1)
    y = np.sin(X * 2 * np.pi) + 3 + np.random.randn(M, 1) * 0.05
    plt.plot(X, y, "b.", label="Data points")
    plt.pause(1)
    taus = np.logspace(np.log10(0.0001), np.log10(1), num=20)  # 从 0.01 到 sqrt(10) 分10个值

    x0 = np.linspace(0, 1, 100).reshape(-1, 1)  # 预测点
    for tau in taus:
        # Locally Weighted Linear Regression
        def predict(X, y, x0, tau):
            m = X.shape[0]
            W = np.diag(np.exp(-np.sum((X - x0)**2, axis=1) / (2 * tau**2)))
            X_b = np.c_[np.ones((m, 1)), X]
            theta = np.linalg.pinv(X_b.T @ W @ X_b) @ X_b.T @ W @ y
            return np.array([1, x0[0]]) @ theta

        y_pred = np.array([predict(X, y, x0_i, tau) for x0_i in x0])
        plt.cla()
        plt.plot(X, y, "b.", label="Data points")
        plt.plot(x0, y_pred, label=f'Tau={tau}')
        plt.title(f"Locally Weighted Regression (Tau = {tau:.4f})")
        plt.pause(1)