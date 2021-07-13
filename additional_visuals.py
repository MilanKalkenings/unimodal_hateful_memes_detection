import numpy as np
import matplotlib.pyplot as plt

# visualization of the exact matcher results
acc_50_50 = np.array([0.59, 0.79, 0.9, 0.93, 0.91, 0.92, 0.91, 0.87, 0.81, 0.72, 0.61, 0.52, 0.5])
recall_50_50 = np.array([0.18, 0.59, 0.83, 0.93, 0.94, 0.98, 0.99, 0.99, 1, 1, 1, 1, 1])
precision_50_50 = np.array([0.96, 0.96, 0.96, 0.93, 0.89, 0.88, 0.85, 0.8, 0.73, 0.64, 0.56, 0.51, 0.5])

x = np.arange(start=1, stop=40, step=3)

fig, ax = plt.subplots(figsize=(8, 8))
fig.suptitle("Balanced Dataset")
plt.plot(x, acc_50_50, label="Accuracy")
plt.plot(x, recall_50_50, label="Recall")
plt.plot(x, precision_50_50, label="Precision")
ax.legend()
ax.set_xlabel("Threshold")
plt.savefig("50_50.png")

acc = np.array([0.95, 0.95, 0.94, 0.91, 0.89, 0.87, 0.82, 0.77, 0.65, 0.48, 0.26, 0.11, 0.08])
recall = np.array([0.16, 0.6, 0.85, 0.91, 0.95, 0.99, 0.99, 0.99, 1, 1, 1, 1, 1])
precision = np.array([0.46, 0.66, 0.56, 0.5, 0.4, 0.36, 0.29, 0.24, 0.17, 0.12, 0.09, 0.07, 0.07])

fig, ax = plt.subplots(figsize=(8, 8))
fig.suptitle("Imbalanced Dataset")
plt.plot(x, acc, label="Accuracy")
plt.plot(x, recall, label="Recall")
plt.plot(x, precision, label="Precision")
ax.legend()
ax.set_xlabel("Threshold")
plt.savefig("full.png")