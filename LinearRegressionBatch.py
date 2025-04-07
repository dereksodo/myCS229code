import matplotlib.pyplot as plt
import numpy as np
import time  # Import the time module

#this program uses batch gradient descent

def drawgraph(theta, X, y, X_b, ax): # Pass the axis object
    ax.cla()  # Clear the previous plot
    #plot the graph
    ax.plot(X, y, "b.")
    #plot the line
    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict = X_new_b.dot(theta)
    ax.plot(X_new, y_predict, "r-")
    ax.axis([0, 2, 0, 15])
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Linear Regression")
    plt.pause(0.01)  # Pause for 0.01 seconds
    return

if __name__ == "__main__":
    # Generate some data
    np.random.seed(0)
    M = 100
    X = 2 * np.random.rand(M, 1)
    y = 4 + 3 * X + np.random.randn(M, 1)

    tolerance = 1e-5

    # Add bias term
    X_b = np.c_[np.ones((M, 1)), X]

    # Initialize theta
    theta = np.zeros_like(X_b, float).T
    learning_rate = 0.01
    # learning_rate cannot be too high or the program fails

    count = 0
    # Gradient descent
    fig, ax = plt.subplots() # Create a figure and an axes object
    while True:
        count += 1
        gradients = 2 / M * X_b.T.dot(X_b.dot(theta) - y)

        # Update theta
        theta -= learning_rate * gradients

        drawgraph(theta, X, y, X_b, ax) # Pass the axes object to the function

        if np.linalg.norm(gradients) < tolerance:
            break
    print("Count = ", count)
    plt.show()

