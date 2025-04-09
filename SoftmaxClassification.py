import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from matplotlib.ticker import FuncFormatter
import numpy as np

def softmax_loss_vectorized(W, X, y, reg):
    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    softmax_output = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    correct_class_score = softmax_output[np.arange(num_train), y]
    loss = -np.sum(np.log(correct_class_score)) / num_train
    loss += reg * np.sum(W * W)

    softmax_output[np.arange(num_train), y] -= 1
    dW = X.T.dot(softmax_output) / num_train + 2 * reg * W
    return loss, dW

if __name__ == "__main__":
    # Example usage
    num_classes = 3
    num_features = 2  # 2D for visualization
    num_samples = 100

    # Randomly generated data
    X = np.zeros((num_samples, num_features))
    y = np.zeros(num_samples, dtype=int)
    radius = 5
    samples_per_class = num_samples // num_classes

    for i in range(num_classes):
        angle = 2 * np.pi * i / num_classes
        center = np.array([np.cos(angle), np.sin(angle)]) * radius
        idx = range(i * samples_per_class, (i + 1) * samples_per_class)
        X[idx] = center + np.random.randn(samples_per_class, num_features) * 2.5
        y[idx] = i

    W = np.random.randn(num_features, num_classes) * 0.01
    reg = 0.1
    learning_rate = 1.0

    for i in range(200):
        loss, dW = softmax_loss_vectorized(W, X, y, reg)
        W -= learning_rate * dW

    # Visualize prediction probability distribution
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = np.exp(grid.dot(W)) / np.sum(np.exp(grid.dot(W)), axis=1, keepdims=True)
    Z = np.argmax(probs, axis=1).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.tab10)
    colors = [plt.cm.tab10(i) for i in y]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=colors, edgecolors='k')

    for i in range(num_classes):
        ax.scatter([], [], c=plt.cm.tab10(i), label=f"Class {i}")
    ax.legend()

    annot = ax.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind, mx, my):
        point = np.array([[mx, my]])
        p = np.exp(point.dot(W)) / np.sum(np.exp(point.dot(W)), axis=1, keepdims=True)
        text = "\n".join([f"Class {i}: {prob:.2f}" for i, prob in enumerate(p[0])])
        annot.xy = (mx, my)
        annot.set_text(text)
        annot.get_bbox_patch().set_alpha(0.8)

    def hover(event):
        if event.inaxes == ax:
            mx, my = event.xdata, event.ydata
            update_annot(None, mx, my)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            annot.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.title("Softmax Multi-class Classification (2D)")
    plt.show()