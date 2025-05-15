import numpy as np
import arff
import os
from SupportVectorMachines import load_arff_data, find_dic_size

class NaiveBayesClassifier:
    def fit(self, X, y):
        m, n = X.shape
        print(f"m = {m}, n = {n}")

        self.classes = np.unique(y)
        self.phi_y = {}
        self.phi = {}

        for cls in self.classes:
            X_cls = X[y == cls]
            self.phi_y[cls] = X_cls.shape[0] / m  # P(y)
            print(f"cls = {cls}, phi_y[cls] = {self.phi_y[cls]}")
            word_counts = np.sum(X_cls, axis=0)   # total count of word j for class y=cls
            total_words = np.sum(word_counts)
            # Laplace smoothing
            self.phi[cls] = (word_counts + 1) / (total_words + n)  # P(x_j | y=cls)
            print(f"cls = {cls}, phi[cls] = {self.phi[cls]}")

    def predict(self, X):
        results = []
        for row in X:
            log_probs = {}
            for cls in self.classes:
                log_prob = np.log(self.phi_y[cls])  # log P(y)
                log_prob += np.sum(row * np.log(self.phi[cls]))  # sum_j x_j * log P(x_j | y)
                log_probs[cls] = log_prob
            results.append(max(log_probs, key=log_probs.get))
        return results

def naive_bayes_predict(train_file, test_file):
    dic_size = find_dic_size(train_file, test_file)
    X_train, y_train = load_arff_data(train_file, dic_size)
    print(y_train)
    X_test, _ = load_arff_data(test_file, dic_size)  # we only need X_test for prediction
    print("train path = ", train_file)
    model = NaiveBayesClassifier()
    model.fit(X_train, y_train)
    return model.predict(X_test)

if __name__ == "__main__":
    from runSVM import load_labels_from_arff
    # Example usage
    train_file = "/Users/tianhaozhang/Downloads/MachineLearning/materials/aimlcs229/q4/train/spam_train_10.arff"
    test_path = "/Users/tianhaozhang/Downloads/MachineLearning/materials/aimlcs229/q4/test/spam_test.arff"
    labels = load_labels_from_arff(test_path)
    predictions = naive_bayes_predict(train_file, test_path)
    suma = 0
    for i in range(len(labels)):
        suma += (predictions[i] == labels[i])
    acc = suma / len(labels)
    print("Accuracy: %.2f" %(acc))