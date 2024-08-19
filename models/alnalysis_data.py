import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PointNet import PointNet

from loader import get_data_for_model
from training_models3 import start_training

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_points_train = 100_000
num_points_val = 50_000
batch_size = 10
num_points_per_cloud = 1024
num_classes = 9

num_batches_train = num_points_train // (num_points_per_cloud * batch_size)
num_batches_val = num_points_val // (num_points_per_cloud * batch_size)
print('Number of batches train:', num_batches_train)
print('Number of batches val:', num_batches_val)

# Получение данных
(data_train_val, cloud_labels_train_val,
 data_test, cloud_labels_test) = get_data_for_model(synthetic=False,
                                                    num_points_train=num_points_train,
                                                    num_points_val=num_points_val,
                                                    num_points_per_cloud=num_points_per_cloud,
                                                    num_classes=num_classes,
                                                    batch_size=batch_size)

# **Новый блок: Анализ распределения классов в обучающей выборке**
unique_classes, counts = np.unique(cloud_labels_train_val, return_counts=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=unique_classes, y=counts, palette="viridis")
plt.xlabel('Класс')
plt.ylabel('Количество примеров')
plt.title('Распределение количества примеров по классам в обучающей выборке')
plt.xticks(ticks=range(num_classes), labels=range(num_classes))
plt.show()

# **Аналогичный анализ для тестовой выборки**
unique_classes_test, counts_test = np.unique(cloud_labels_test, return_counts=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=unique_classes_test, y=counts_test, palette="magma")
plt.xlabel('Класс')
plt.ylabel('Количество примеров')
plt.title('Распределение количества примеров по классам в тестовой выборке')
plt.xticks(ticks=range(num_classes), labels=range(num_classes))
plt.show()