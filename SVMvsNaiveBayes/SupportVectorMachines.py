import numpy as np
import arff
from pathlib import Path
from SequentialMinimalOptimization import sequentialMinimalOptimization

def find_dic_size(file_path1, file_path2):
    with open(file_path1, 'r') as f:
        dataset1 = arff.load(f)
    with open(file_path2, 'r') as f:
        dataset2 = arff.load(f)

    return max(len(dataset1['attributes']), len(dataset2['attributes'])) - 1

def load_arff_data(file_path, dic_size):
    with open(file_path, 'r') as f:
        dataset = arff.load(f)

    data = dataset['data']
    class_index = len(dataset['attributes']) - 1
    num_features = dic_size

    X = np.zeros((len(data), num_features))
    y = []

    is_sparse = isinstance(data[0], list) and isinstance(data[0][0], (tuple, list))

    for i, row in enumerate(data):
        if is_sparse:
            for entry in row:
                idx, val = entry
                if idx == class_index and (val == 'spam' or val == 'non_spam'):
                    y.append(1 if val == 'spam' else -1)
                else:
                    X[i, idx] = val
        else:
            row_data = [float(v) for v in row[:-1]]
            X[i, :len(row_data)] = row_data
            label = row[-1].decode() if isinstance(row[-1], bytes) else row[-1]
            y.append(1 if label == 'spam' else -1)

    return X, np.array(y)

def svm_predict(train_file, test_file, C=1.0, tol=1e-3, max_passes=5, kernel='guassian'):
    dic_size = find_dic_size(train_file, test_file)
    X_train, y_train = load_arff_data(train_file, dic_size)
    X_test, _ = load_arff_data(test_file, dic_size)  # we only need X_test for prediction
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    smo_model = sequentialMinimalOptimization(X_train, y_train, C=C, tol=tol, max_passes=max_passes, kernel=kernel)
    predictions = smo_model.predict(X_test)
    return predictions
