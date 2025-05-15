import numpy as np
inf = 1e18
class sequentialMinimalOptimization:
    def __init__(self, X, y, C=1.0, tol=1e-3, max_passes=5, kernel='linear'):
        self.X = X  # Store the input features matrix
        self.y = y  # Store the target labels vector
        self.C = C  # Regularization parameter controlling trade-off between margin and misclassification
        self.tol = tol  # Tolerance for stopping criterion in optimization
        self.max_passes = max_passes  # Maximum number of passes over data without alpha changes before stopping
        self.m = X.shape[0]  # Number of training samples
        self.alphas = np.zeros(self.m)  # Initialize Lagrange multipliers (alphas) to zero
        self.b = 0  # Initialize bias term to zero
        self.vis = np.full((self.m, self.m), -inf)  # Initialize kernel cache with -inf to indicate uncomputed values
        if kernel == 'linear': 
            self.kernel = self.linear_kernel  # Assign linear kernel function
        elif kernel == 'gaussian': 
            self.kernel = self.guassian_kernel  # Assign Gaussian kernel function
        elif kernel == 'polynomial': 
            self.kernel = self.polynomial_kernel  # Assign polynomial kernel function
        else: 
            self.kernel = self.linear_kernel  # Default to Gaussian kernel if unknown type is provided
        self.edgeIndex = []  # Initialize edge index to track samples on the margin
        self.E = np.zeros(self.m)  # Initialize error cache to store prediction errors for each sample
        self.min_index = 0
        self.max_index = 0
        self.fit()  # Start training the SVM model using SMO algorithm
    
    def K(self, i, j): # reduce repeated calculations of Kernel
        if self.vis[i][j] == -inf:
            self.vis[i][j] = self.kernel(self.X[i],self.X[j])
            return self.vis[i][j]  # Compute kernel value between samples i and j
        else:
            return self.vis[i][j]
        #return self.kernel(self.X[i],self.X[j])

    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2)  # Compute linear kernel as dot product of two vectors
    
    def guassian_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2) ** 2)  # Compute Gaussian kernel (RBF) as exponential of negative squared distance
    
    def polynomial_kernel(self, x1, x2, p=2):
        return (1 + np.dot(x1, x2)) ** p  # Compute polynomial kernel as (1 + dot product) raised to power p
    
    def compute_E(self, i):
        return self.predict_one(self.X[i]) - self.y[i] # Calculate error for i-th training sample (difference between prediction and true label)

    def predict_one(self, x):
        # returns w^T * x - b
        # Compute the decision function output for a single input x by summing over all support vectors weighted by alpha and label, plus bias
        result = sum(self.alphas[j] * self.y[j] * self.kernel(self.X[j], x) for j in range(self.m)) - self.b
        return result  # Return the raw prediction score
        # why going through all support vectors instead of just the support vectors?
        # because we are not storing the support vectors separately, we are using all the data points
        # this may be inefficient, but it is a simple implementation

    def predict(self, X_test):
        # Predict labels for a set of test samples by applying predict_one and thresholding at zero
        return [1 if self.predict_one(x) >= 0 else -1 for x in X_test]

    def update(self, i, j, diffi, diffj):
        self.E[i] = self.compute_E(i)
        self.E[j] = self.compute_E(j)  # Update error cache for samples i and j after alpha updates
        for k in range(self.m):
            if k != i and k != j:
                self.E[k] += diffi * self.y[i] * self.K(i,k) + diffj * self.y[j] * self.K(j,k)  # Update error cache for other samples based on changes in alphas
                #self.E[k] = self.compute_E(k)  # Recompute error for other samples
        self.edgeIndex = []
        self.min_index = self.max_index = i
        # Update edgeIndex based on the new alpha values
        for k in range(self.m):
            if self.alphas[k] == 0 or self.alphas[k] == self.C:
                    self.edgeIndex.append(k)
                    if(self.E[k] < self.E[self.min_index]):
                        self.min_index = k
                    if(self.E[k] >= self.E[self.max_index]):
                        self.max_index = k
        return

    def take_step(self, i, j):
        #print(i,j)
        # Check if i and j are the same, if so, return without doing anything
        if i == j:
            return 0

        # Save old alpha values for i and j
        alpha_i_old = self.alphas[i]
        alpha_j_old = self.alphas[j]

        if self.y[i] != self.y[j]:
            L = max(0, self.alphas[j] - self.alphas[i])  # Lower bound for alpha j
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])  # Upper bound for alpha j
        else:
            L = max(0, self.alphas[i] + self.alphas[j] - self.C)  # Lower bound for alpha j
            H = min(self.C, self.alphas[i] + self.alphas[j])  # Upper bound for alpha j
        if L == H:
            return 0  # If bounds are equal, no update possible, skip to next i

        # Compute eta, the second derivative of the objective function along the direction of alpha j
        eta = -2 * self.K(i,j) + self.K(i,i) + self.K(j,j)
        if eta > 0:
            # Update alpha j using the computed errors and eta
            self.alphas[j] += self.y[j] * (self.E[i] - self.E[j]) / eta
            self.alphas[j] = np.clip(self.alphas[j], L, H)  # Clip alpha j to be within bounds L and H
            diffj = self.alphas[j] - alpha_j_old
        else:
            return 0

        if abs(diffj) < 1e-5:
            return 0  # If change in alpha j is negligible, skip further updates

        # Update alpha i in opposite direction to maintain constraints
        self.alphas[i] -= self.y[i] * self.y[j] * diffj
        diffi = self.alphas[i] - alpha_i_old  # Compute change in alpha i

        # Compute bias terms b1 and b2 based on updated alphas and errors
        b1 = self.b + self.E[i] + self.y[i] * diffi * self.K(i,i) + self.y[j] * diffj * self.K(i,j)
        b2 = self.b + self.E[j] + self.y[i] * diffi * self.K(i,j) + self.y[j] * diffj * self.K(j,j)

        # Update bias term b based on whether alpha i or j is within bounds or average otherwise
        if 0 < self.alphas[i] < self.C:
            self.b = b1
        elif 0 < self.alphas[j] < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        self.update(i, j, diffi, diffj)  # Update error cache with new alphas
        return 1  # Return 1 to indicate that an update was made

    def examine_example(self, i):
        # Compute error for the i-th training sample
        self.E[i] = self.compute_E(i)
        # Check if sample violates KKT conditions and is eligible for optimization
        if (self.y[i] * self.E[i] < -self.tol and self.alphas[i] < self.C) or (self.y[i] * self.E[i] > self.tol and self.alphas[i] > 0):
            # Heuristic: choose choice j to maximize |E_i - E_j|, where E_j is the error for sample j
            if self.min_index != self.max_index:
                if(self.E[i] > 0):
                    j = self.min_index
                else:
                    j = self.max_index
                if self.take_step(i, j) == 1:
                    return 1
            length = len(self.edgeIndex)
            for j in range(length):
                if self.take_step(i, j) == 1:
                    return 1
            for j in range(self.m):
                if self.take_step(i, j) == 1:
                    return 1     
        return 0

    def fit(self):
        numChanged = 0
        examineAll = 1
        while numChanged > 0 or examineAll:
            numChanged = 0
            if examineAll:
                for i in range(self.m):
                    numChanged += self.examine_example(i)
            else:
                length = len(self.edgeIndex)
                for i in range(length):
                        numChanged += self.examine_example(i)
            if examineAll == 1:
                examineAll = 0
            elif numChanged == 0:
                examineAll = 1



        passes = 0  # Initialize count of passes without any alpha updates
        while passes < self.max_passes:  # Loop until max passes without alpha changes reached
            alpha_changed = 0  # Counter for number of alpha pairs updated in this pass
            for i in range(self.m):  # Iterate over all training samples
                #self.E[i] = self.compute_E(i)
                # Check if sample violates KKT conditions and is eligible for optimization
                if (self.y[i] * self.E[i] < -self.tol and self.alphas[i] < self.C) or (self.y[i] * self.E[i] > self.tol and self.alphas[i] > 0):
                    # Heuristic: choose choice j to maximize |E_i - E_j|, where E_j is the error for sample j
                    j = i
                    tmp = i
                    for _ in range(self.m):
                        if (abs(self.E[j] - self.E[i]) < abs(self.E[i] - self.E[_])):
                            j = _
                    if j == i:
                        j = np.random.choice([x for x in range(self.m) if x != i])
                    #self.E[j] = self.compute_E(j)

                    alpha_i_old = self.alphas[i]  # Save old alpha i value for reference
                    alpha_j_old = self.alphas[j]  # Save old alpha j value for reference

                    # Compute bounds L and H for alpha j based on whether labels i and j are same or different
                    if self.y[i] != self.y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])  # Lower bound for alpha j
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])  # Upper bound for alpha j
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)  # Lower bound for alpha j
                        H = min(self.C, self.alphas[i] + self.alphas[j])  # Upper bound for alpha j
                    if L == H:
                        continue  # If bounds are equal, no update possible, skip to next i

                    # Compute eta, the second derivative of the objective function along the direction of alpha j
                    eta = -2 * self.K(i,j) + self.K(i,i) + self.K(j,j)
                    if eta <= 0:
                        continue

                    # Update alpha j using the computed errors and eta
                    self.alphas[j] += self.y[j] * (self.E[i] - self.E[j]) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)  # Clip alpha j to be within bounds L and H
                    diffj = self.alphas[j] - alpha_j_old
                    if abs(diffj) < 1e-5:
                        continue  # If change in alpha j is negligible, skip further updates

                    # Update alpha i in opposite direction to maintain constraints
                    self.alphas[i] -= self.y[i] * self.y[j] * diffj
                    diffi = self.alphas[i] - alpha_i_old  # Compute change in alpha i

                    # Compute bias terms b1 and b2 based on updated alphas and errors
                    b1 = self.b + self.E[i] + self.y[i] * diffi * self.K(i,i) + self.y[j] * diffj * self.K(i,j)
                    b2 = self.b + self.E[j] + self.y[i] * diffi * self.K(i,j) + self.y[j] * diffj * self.K(j,j)

                    # Update bias term b based on whether alpha i or j is within bounds or average otherwise
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    self.update(i, j, diffi, diffj)
                    #self.update(i, j, diffi, diffj)  # Update error cache with new alphas
                    alpha_changed += 1  # Increment count of alpha pairs updated
            if alpha_changed == 0:
                passes += 1  # Increment passes count if no alphas were updated
            else:
                passes = 0  # Reset passes count if any alpha was updated in this iteration