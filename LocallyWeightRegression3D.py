import matplotlib.pyplot as plt
import numpy as np
import math
import time
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
# KFold & mean_squared_error are used for cross-validation and error calculation
# Locally Weighted Linear Regression
if __name__ == "__main__":
    # Generate some data
    np.random.seed(0)
    M = 100
    house_size = np.random.uniform(500, 3500, (M, 1))
    num_rooms = np.random.randint(1, 11, (M, 1))
    X = np.hstack((house_size, num_rooms))
    y = 100 + 0.1 * house_size + 10 * num_rooms + np.random.randn(M, 1) * 10000
    plt.plot(X[:, 0], y, "b.", label="Data points")
    plt.pause(1)
    taus = np.logspace(np.log10(0.01), np.log10(1000), num=20)

    best_tau = None
    lowest_mse = float('inf')

    s = np.linspace(500, 3500, 50)
    r = np.linspace(1, 10, 50)
    S, R = np.meshgrid(s, r)
    x0 = np.c_[S.ravel(), R.ravel()]
    for tau in taus:
        # Locally Weighted Linear Regression
        def predict(X, y, x0, tau):
            m = X.shape[0]
            w_raw = np.exp(-np.sum((X - x0)**2, axis=1) / (2 * tau**2))
            w_raw = np.clip(w_raw, 1e-8, 1)  # Prevent overflow and underflow
            W = np.diag(w_raw)
            X_b = np.c_[np.ones((m, 1)), X]
            theta = np.linalg.pinv(X_b.T @ W @ X_b) @ X_b.T @ W @ y
            return np.r_[1, x0] @ theta

        try:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            mse_list = []
            for train_index, val_index in kf.split(X):
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                y_val_pred = np.array([predict(X_train, y_train, x_i, tau) for x_i in X_val])
                mse_fold = mean_squared_error(y_val, y_val_pred)
                mse_list.append(mse_fold)
            mse = np.mean(mse_list)
        except Exception as e:
            print(f"Tau = {tau:.4f} 出现错误: {e}")
            continue
        print(f"Tau = {tau:.4f}, MSE = {mse:.2f}")
        if mse < lowest_mse:
            lowest_mse = mse
            best_tau = tau

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.cla()
        ax.scatter(X[:, 0], X[:, 1], y, c='b', label="Data points")
        y_pred = np.array([predict(X, y, x0_i, tau) for x0_i in x0])
        ax.plot_trisurf(x0[:, 0], x0[:, 1], y_pred.ravel(), cmap='viridis', alpha=0.7)
        ax.set_xlabel("House Size")
        ax.set_ylabel("Number of Rooms")
        ax.set_zlabel("Price")
        ax.set_title(f"Locally Weighted Regression (Tau = {tau:.4f})")
        plt.pause(1)
        plt.close(fig)

    print(f"\n✅ Best Tau: {best_tau:.4f} with MSE = {lowest_mse:.2f}")