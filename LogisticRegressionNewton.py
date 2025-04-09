import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Generate some data
    np.random.seed(0)
    M = 100
    X = np.random.rand(M, 1)

    # Generate labels with some noise with a bias of 0.2
    true_weights = np.array([[5]])
    bias = -2.5
    logits = X @ true_weights + bias
    prob = 1 / (1 + np.exp(-logits))
    y = (prob > np.random.rand(M, 1) + 0.2).astype(int)

    # Define sigmoid function
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    # Train logistic regression using Newton's method
    def train_logistic_regression(X, y, epochs=10):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.zeros((X_b.shape[1], 1))
        for epoch in range(epochs):
            logits = X_b @ theta
            predictions = sigmoid(logits)
            gradient = X_b.T @ (predictions - y)
            S = np.diag((predictions * (1 - predictions)).ravel())
            H = X_b.T @ S @ X_b
            theta -= np.linalg.inv(H) @ gradient
            
            if epoch % 1 == 0:
                plt.clf()
                x_vals = np.linspace(0, 1, 100).reshape(-1, 1)
                x_vals_b = np.c_[np.ones((x_vals.shape[0], 1)), x_vals]
                y_vals = sigmoid(x_vals_b @ theta)
                plt.scatter(X, y, c=y.ravel(), cmap="bwr", label="Data")
                plt.plot(x_vals, y_vals, 'k-', label=f"Epoch {epoch}")
                plt.xlabel("X")
                plt.ylabel("Probability")
                plt.title(f"Logistic Regression (Newton's Method) Epoch {epoch}")
                plt.legend()
                plt.pause(0.1)
        return theta

    plt.ion()  # Turn on interactive mode
    # Train the model
    theta = train_logistic_regression(X, y)
    plt.ioff()  # Turn off interactive mode

    # Plot the data and the decision boundary
    plt.figure()
    plt.scatter(X, y, c=y.ravel(), cmap="bwr", label="Data")
    x_vals = np.linspace(0, 1, 100).reshape(-1, 1)
    x_vals_b = np.c_[np.ones((x_vals.shape[0], 1)), x_vals]
    y_vals = sigmoid(x_vals_b @ theta)
    plt.plot(x_vals, y_vals, 'k-', label="Logistic curve")
    plt.xlabel("X")
    plt.ylabel("Probability")
    plt.title("Logistic Regression")
    plt.legend()
    plt.show()