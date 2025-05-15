import numpy as np
import matplotlib.pyplot as plt

class GDA:
    def __init__(self):
        self.phi = None
        self.mu_0 = None
        self.mu_1 = None
        self.sigma = None
        self.sigma_inv = None
        self.theta = None

    def fit(self, X, y):
        m, n = X.shape

        # 计算phi, mu_0, mu_1
        self.phi = np.mean(y)
        self.mu_0 = np.mean(X[y == 0], axis=0)
        self.mu_1 = np.mean(X[y == 1], axis=0)
        print(self.mu_0, self.mu_1)

        # 计算共享的协方差矩阵sigma
        sigma = np.zeros((n, n))
        for i in range(m):
            xi = X[i].reshape(-1, 1)
            if y[i] == 0:
                mu = self.mu_0.reshape(-1, 1)
            else:
                mu = self.mu_1.reshape(-1, 1)
            sigma += (xi - mu).dot((xi - mu).T)

        self.sigma = sigma / m
        self.sigma_inv = np.linalg.inv(self.sigma)

    def predict(self, X):
        # 计算P(y=1|x)和P(y=0|x)哪个大
        # P(y=1|x) = P(x|y=1) * P(y=1) / P(x)
        # P(y=0|x) = P(x|y=0) * P(y=0) / P(x)
        # 这里我们只需要计算P(y=1|x)和P(y=0|x)的比值
        # 计算P(x|y=1)和P(x|y=0)
        # P(x|y=1) = N(x; mu_1, sigma)
        # P(x|y=0) = N(x; mu_0, sigma)
        # 计算P(x|y=1)和P(x|y=0)的大小差距
        K = (X - self.mu_1).dot(self.sigma_inv).dot((X - self.mu_1).T) - (X - self.mu_0).dot(self.sigma_inv).dot((X - self.mu_0).T)
        return 1 - K

if __name__ == "__main__":
    # 生成测试数据
    np.random.seed(0)
    m = 100
    X_class0 = np.random.multivariate_normal([1, 1], [[1, 0.5], [0.5, 1]], m)
    X_class1 = np.random.multivariate_normal([3, 3], [[1, -0.5], [-0.5, 1]], m)

    X = np.vstack((X_class0, X_class1))
    y = np.hstack((np.zeros(m), np.ones(m)))

    # 训练模型
    model = GDA()
    model.fit(X, y)

    # 预测
    y_pred = model.predict(X)

    # 计算准确率
    accuracy = np.mean(y_pred == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # 可视化结果
    plt.scatter(X_class0[:, 0], X_class0[:, 1], label="Class 0", alpha=0.6)
    plt.scatter(X_class1[:, 0], X_class1[:, 1], label="Class 1", alpha=0.6)


    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Gaussian Discriminant Analysis (GDA)')
    plt.grid(True)

    # 初始化用于显示预测点的散点图对象
    pred_point, = plt.plot([], [], 'o', markersize=8)

    def on_mouse_move(event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            point = np.array([[x, y]])
            prediction = model.predict(point)
            pred_label = 1 if prediction >= 0.5 else 0
            color = 'red' if pred_label == 1 else 'blue'
            pred_point.set_data([x], [y])
            pred_point.set_color(color)
            plt.gcf().canvas.draw_idle()

    plt.connect('motion_notify_event', on_mouse_move)
    plt.show()