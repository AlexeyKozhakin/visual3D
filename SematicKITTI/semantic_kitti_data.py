import os
import torch
from torch.utils.data import Dataset
import struct


# Загрузка данных SemanticKITTI
def load_semantickitti_data(sequence, partition):
    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = r'C:\Users\alexe\Downloads\UM\work\Data\Semantic KITTI\Semantic KITTI\dataset\sequences'
    # DATA_DIR = os.path.join(BASE_DIR, sequence)
    DATA_DIR = BASE_DIR
    point_clouds = []
    labels_clouds = []

    # Для тренировочной и тестовой части загружаем разные последовательности
    if partition == 'train':
        sequences = [4]  # Выбор тренировочных последовательностей
    elif partition == 'test':
        sequences = [6]  # Выбор тестовых последовательностей
    else:
        raise ValueError("Unknown partition. Choose from 'train' or 'test'.")

    # Загрузка данных для каждой последовательности
    for seq in sequences:
        seq_dir = os.path.join(DATA_DIR, f'{seq:02d}', 'velodyne')
        label_dir = os.path.join(DATA_DIR, f'{seq:02d}', 'labels')

        point_files = sorted(os.listdir(seq_dir))
        label_files = sorted(os.listdir(label_dir))

        # Загрузка каждой точки облака
        for point_file, label_file in zip(point_files, label_files):
            point_path = os.path.join(seq_dir, point_file)
            label_path = os.path.join(label_dir, label_file)

            # Загрузка точек из файла .bin
            points = np.fromfile(point_path, dtype=np.float32).reshape(-1, 4)  # (x, y, z, intensity)
            labels = np.fromfile(label_path, dtype=np.uint32).reshape(-1)  # метки классов
            labels = labels & 0xFFFF  # Применение маски для извлечения меток классов

            point_clouds.append(points)
            labels_clouds.append(labels)

    return point_clouds, labels_clouds


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud[:, :3], xyz1), xyz2).astype('float32')
    pointcloud[:, :3] = translated_pointcloud
    return pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    jitter = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    pointcloud[:, :3] += jitter[:, :3]  # Применение джиттера только к координатам XYZ
    return pointcloud


class SemanticKITTIDataset(Dataset):
    def __init__(self, num_points, partition='train', sequence='00'):
        self.num_points = num_points
        self.partition = partition
        self.point_paths, self.label_paths = self.get_file_paths(sequence, partition)

    def get_file_paths(self, sequence, partition):
        BASE_DIR = r'C:\Users\alexe\Downloads\UM\work\Data\Semantic KITTI\Semantic KITTI\dataset\sequences'
        point_paths = []
        label_paths = []

        # Выбираем последовательности для тренировочной или тестовой части
        if partition == 'train':
            sequences = [4]  # Тренировочные последовательности
        elif partition == 'test':
            sequences = [6]  # Тестовые последовательности
        else:
            raise ValueError("Unknown partition. Choose from 'train' or 'test'.")

        # Сохраняем пути к каждому файлу облаков точек и меток
        for seq in sequences:
            seq_dir = os.path.join(BASE_DIR, f'{seq:02d}', 'velodyne')
            label_dir = os.path.join(BASE_DIR, f'{seq:02d}', 'labels')

            point_files = sorted(os.listdir(seq_dir))
            label_files = sorted(os.listdir(label_dir))

            point_paths += [os.path.join(seq_dir, f) for f in point_files]
            label_paths += [os.path.join(label_dir, f) for f in label_files]

        return point_paths, label_paths

    def __getitem__(self, index):
        point_path = self.point_paths[index]
        label_path = self.label_paths[index]

        # Загружаем данные для конкретного индекса
        pointcloud = np.fromfile(point_path, dtype=np.float32).reshape(-1, 4)  # (x, y, z, intensity)
        label = np.fromfile(label_path, dtype=np.uint32).reshape(-1)  # Метки классов
        label = label & 0xFFFF  # Применение маски для извлечения меток классов

        # Ограничение числа точек
        if len(pointcloud) > self.num_points:
            choice = np.random.choice(len(pointcloud), self.num_points, replace=False)
        else:
            choice = np.random.choice(len(pointcloud), self.num_points, replace=True)

        pointcloud = pointcloud[choice]
        label = label[choice]

        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)

        # Преобразуем метки в поддерживаемый тип (например, int32)
        label = torch.tensor(label.astype(np.int32), dtype=torch.int64)

        return torch.tensor(pointcloud, dtype=torch.float32), label

    def __len__(self):
        return len(self.point_paths)


import os
import numpy as np
from collections import Counter


def compute_class_distribution(sequence, partition='train'):
    # Путь к данным
    BASE_DIR = r'C:\Users\alexe\Downloads\UM\work\Data\Semantic KITTI\Semantic KITTI\dataset\sequences'
    DATA_DIR = BASE_DIR
    class_counter = Counter()

    # Выбор последовательностей для анализа
    if partition == 'train':
        sequences = [4]  # Последовательности для тренировки
    elif partition == 'test':
        sequences = [6]  # Последовательности для тестирования
    else:
        raise ValueError("Unknown partition. Choose from 'train' or 'test'.")

    # Проход по каждой последовательности
    for seq in sequences:
        seq_dir = os.path.join(DATA_DIR, f'{seq:02d}', 'velodyne')
        label_dir = os.path.join(DATA_DIR, f'{seq:02d}', 'labels')

        point_files = sorted(os.listdir(seq_dir))
        label_files = sorted(os.listdir(label_dir))

        # Обработка файлов
        for label_file in label_files:
            label_path = os.path.join(label_dir, label_file)

            # Загружаем метки классов из .bin файла
            labels = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
            labels = labels & 0xFFFF  # Применение маски для извлечения меток классов

            # Обновляем счетчик классов
            class_counter.update(labels)

    return class_counter


def compute_class_percentages(class_counter):
    total_count = sum(class_counter.values())
    class_percentages = {cls: (count / total_count) * 100 for cls, count in class_counter.items()}
    return class_percentages


# Пример использования
if __name__ == '__main__':
    class_counter = compute_class_distribution(sequence='00', partition='train')
    class_percentages = compute_class_percentages(class_counter)

    print("Распределение классов:")
    for class_id, count in class_counter.items():
        percentage = class_percentages[class_id]
        print(f"Class {class_id}: {count} instances ({percentage:.2f}%)")

# Пример использования
if __name__ == '__main__' and 1==2:
    # Инициализация тренировочного и тестового датасетов
    train_dataset = SemanticKITTIDataset(num_points=4096, partition='train')
    test_dataset = SemanticKITTIDataset(num_points=4096, partition='test')

    # Тестирование на нескольких примерах из тренировочного датасета
    print("Train dataset:")
    for i in range(5):  # Можно изменить диапазон, чтобы посмотреть больше примеров
        data, label = train_dataset[i]  # Загружаем i-й элемент
        print(f"Example {i + 1}:")
        print("Point cloud shape:", data.shape)
        print("Label shape:", label.shape)
        print("First 5 points in point cloud:\n", data[:5])  # Выводим первые 5 точек
        print("First 5 labels:\n", label[:5])  # Выводим первые 5 меток
        print('-' * 50)

    # Тестирование на нескольких примерах из тестового датасета
    print("Test dataset:")
    for i in range(5):  # Можно изменить диапазон, чтобы посмотреть больше примеров
        data, label = test_dataset[i]  # Загружаем i-й элемент
        print(f"Example {i + 1}:")
        print("Point cloud shape:", data.shape)
        print("Label shape:", label.shape)
        print("First 5 points in point cloud:\n", data[:5])  # Выводим первые 5 точек
        print("First 5 labels:\n", label[:5])  # Выводим первые 5 меток
        print('-' * 50)

