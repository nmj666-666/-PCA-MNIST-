# ==== 1. 加载数据 ====
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# ==== 2. 数据预处理 ====
X = X / 255.0  # 归一化到[0, 1]

# ==== 3. PCA降维 ====
from sklearn.decomposition import PCA
pca = PCA(n_components=50)  # 降至50维
X_pca = pca.fit_transform(X)  # 应用PCA

# 查看保留的方差比例
print("保留方差比例：", sum(pca.explained_variance_ratio_))

# ==== 4. 划分训练集和测试集 ====
from sklearn.model_selection import train_test_split

# 原始数据划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 降维后数据划分
X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# ==== 5. 训练SVM模型 ====
from sklearn.svm import SVC
import time

# 训练原始数据模型
svm_original = SVC(kernel='rbf')
start = time.time()
svm_original.fit(X_train, y_train)
original_time = time.time() - start
original_accuracy = svm_original.score(X_test, y_test)

# 训练PCA降维后数据模型
svm_pca = SVC(kernel='rbf')
start = time.time()
svm_pca.fit(X_pca_train, y_pca_train)
pca_time = time.time() - start
pca_accuracy = svm_pca.score(X_pca_test, y_pca_test)

# 输出结果
print("\n原始数据训练时间：{:.1f}s，准确率：{:.3f}".format(original_time, original_accuracy))
print("PCA后数据训练时间：{:.1f}s，准确率：{:.3f}".format(pca_time, pca_accuracy))

# ==== 6. 可视化 ====
import matplotlib.pyplot as plt
import numpy as np

# 6.1 绘制PCA降维后的2D分布图
plt.figure(figsize=(10, 6))
sample_idx = np.random.choice(len(X_pca), 1000, replace=False)  # 随机选1000个样本
X_plot = X_pca[sample_idx]
y_plot = y[sample_idx].astype(int)

plt.scatter(X_plot[:, 0], X_plot[:, 1], c=y_plot, cmap='tab10', alpha=0.6)
plt.colorbar(label='Digit Class')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("MNIST PCA 2D Projection")
plt.savefig("pca_visualization.png")  # 保存图片
plt.show()

# 6.2 对比训练时间和准确率
labels = ['Original Data', 'PCA Data']
time_values = [original_time, pca_time]
accuracy_values = [original_accuracy, pca_accuracy]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(labels, time_values, color=['blue', 'orange'])
plt.title("Training Time Comparison")
plt.ylabel("Time (s)")

plt.subplot(1, 2, 2)
plt.bar(labels, accuracy_values, color=['blue', 'orange'])
plt.title("Test Accuracy Comparison")
plt.ylabel("Accuracy")

plt.tight_layout()
plt.savefig("comparison.png")
plt.show()