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

(data_train_val, cloud_labels_train_val,
 data_test, cloud_labels_test) = get_data_for_model(synthetic=False,
                                                    num_points_train=num_points_train,
                                                    num_points_val=num_points_val,
                                                    num_points_per_cloud=num_points_per_cloud,
                                                    num_classes=num_classes,
                                                    batch_size=batch_size)

# Шаг 1: Загрузка сохраненной модели
model = PointNet(classes=num_classes)  # Замените на вашу модель, если нужно
model.load_state_dict(torch.load('model.pth'))
model = model.to(device)  # Перемещение модели на GPU или CPU
model.eval()  # Переводим модель в режим оценки (отключение dropout и batch norm)

# Шаг 3: Применение модели к тестовым данным для получения предсказаний
all_preds = []
all_labels = []
# X_train = data_train_val
# y_train = cloud_labels_train_val
X_train = data_test
y_train = cloud_labels_test

with torch.no_grad():
    for i in range(num_batches_train):
        # Перемещение данных на устройство и корректная транспозиция
        X_test_batch = torch.tensor(X_train[i * batch_size:(i + 1) * batch_size], dtype=torch.float32).permute(0, 2, 1).to(device)
        y_test_batch = torch.tensor(y_train[i * batch_size:(i + 1) * batch_size], dtype=torch.long).to(device)

        # Получение предсказаний
        output, _, _ = model(X_test_batch)
        _, predicted = torch.max(output, 1)

        # Перемещение предсказаний на CPU и добавление в список
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_test_batch.cpu().numpy())

# Преобразование списков в numpy массивы
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Шаг 4: Вычисление матрицы ошибок
conf_matrix = confusion_matrix(all_labels, all_preds)

# Шаг 5: Отображение матрицы ошибок
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=range(num_classes),
            yticklabels=range(num_classes))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
