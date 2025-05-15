import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import arff
import re
import time
import pandas as pd
from datetime import datetime

## Prepare lists to store results for both SVM and Naive Bayes
elapsed_svm_list, elapsed_nb_list = [], []
x_vals, svm_accs, nb_accs = [], [], []

plt.ion()
fig, (ax, ax_table) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
line_svm, = ax.plot([], [], 'bo-', label='SVM')
line_nb, = ax.plot([], [], 'ro-', label='Naive Bayes')
ax.set_xlabel("File Number")
ax.set_ylabel("Error (%)")
ax.set_title("SVM vs Naive Bayes Error Percentage by Train File")
ax.set_ylim(0, 15)
ax.set_xlim(right=3000)
ax.grid(True)
ax.legend()

# 添加当前目录到路径，确保可导入本地模块
sys.path.append(os.path.dirname(__file__))

from SupportVectorMachines import svm_predict
from NaiveBayes import naive_bayes_predict

def evaluate(predictions, labels):
    correct = np.sum(predictions == labels)
    return correct / len(labels)

def load_labels_from_arff(file_path):
    with open(file_path, 'r') as f:
        dataset = arff.load(f)
    data = dataset['data']
    y = []
    for i, row in enumerate(data):
        if row[-1] == 'spam':
            y.append(1)
        else:
            y.append(-1)
    return y

def extract_file_number(filename):
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

if __name__ == "__main__":
    # 数据路径
    train_folder = "/Users/tianhaozhang/Downloads/MachineLearning/materials/aimlcs229/q4/train"
    test_path = "/Users/tianhaozhang/Downloads/MachineLearning/materials/aimlcs229/q4/test/spam_test.arff"

    # Sort files based on numbers extracted from filenames
    train_files = sorted([f for f in os.listdir(train_folder) if f.endswith(".arff")], key=extract_file_number)

    print(f"Train files: {train_files}")
    print(f"Test file: {test_path}")
    labels = load_labels_from_arff(test_path)

    ax_table.axis('off')
    table = None

    # Ensure plots directory exists
    os.makedirs("plots", exist_ok=True)

    # Process all train files, store SVM and NB accuracies
    for train_file in train_files:
        if train_file == "spam_train_10.arff":
            continue
        train_path = os.path.join(train_folder, train_file)
        start_svm = time.time()
        predictions_SVM = svm_predict(train_path, test_path)
        elapsed_svm = time.time() - start_svm

        start_nb = time.time()
        predictions_NB = naive_bayes_predict(train_path, test_path)
        elapsed_nb = time.time() - start_nb

        
        acc_SVM = evaluate(np.array(predictions_SVM), labels)
        acc_NB = evaluate(np.array(predictions_NB), labels)

        elapsed_svm_list.append(elapsed_svm)
        elapsed_nb_list.append(elapsed_nb)

        file_num = extract_file_number(train_file)
        x_vals.append(file_num)
        svm_accs.append(acc_SVM)
        nb_accs.append(acc_NB)

        # Compute error percentages
        svm_errors = [(1 - acc) * 100 for acc in svm_accs]
        nb_errors = [(1 - acc) * 100 for acc in nb_accs]

        # Start plotting from the second data point
        plot_x = x_vals[1:] if len(x_vals) > 1 else []
        plot_svm_errors = svm_errors[1:] if len(svm_errors) > 1 else []
        plot_nb_errors = nb_errors[1:] if len(nb_errors) > 1 else []

        line_svm.set_xdata(plot_x)
        line_svm.set_ydata(plot_svm_errors)
        line_nb.set_xdata(plot_x)
        line_nb.set_ydata(plot_nb_errors)
        ax.relim()
        ax.autoscale_view()
        ax.set_ylim(0, 15)
        ax.set_xlim(0, 3000)

        # Clear previous annotations
        for txt in ax.texts:
            txt.set_visible(False)

        for x, err in zip(plot_x, plot_svm_errors):
            ax.annotate(f"{err:.2f}", (x, err), textcoords="offset points", xytext=(0,5), ha='center')
        for x, err in zip(plot_x, plot_nb_errors):
            ax.annotate(f"{err:.2f}", (x, err), textcoords="offset points", xytext=(0,-15), ha='center')

        fig.canvas.draw()
        fig.canvas.flush_events()
        # Save the figure as a vector image (SVG) to a specific folder with timestamp to avoid overwrite
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"svm_error_plot_{timestamp}.svg"
        plt.savefig(os.path.join("plots", filename), format="svg")

        df = pd.DataFrame({
            'File Number': x_vals,
            'SVM Error (%)': [f'{err:.2f}' for err in svm_errors],
            'NB Error (%)': [f'{err:.2f}' for err in nb_errors],
            'SVM Time': [f'{t:.2f}s' for t in elapsed_svm_list],
            'NB Time': [f'{t:.2f}s' for t in elapsed_nb_list]
        })

        ax_table.clear()
        ax_table.axis('off')
        table = ax_table.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
        table.scale(1, 1.5)
        fig.canvas.draw()
        fig.canvas.flush_events()