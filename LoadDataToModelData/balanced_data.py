import numpy as np

def balance_data(X, y):
    # Определяем количество элементов в датасете
    size_data = X.shape[0]

    # Находим уникальные классы и количество классов
    unique_classes, class_counts = np.unique(y, return_counts=True)
    num_classes = len(unique_classes)

    # Определяем максимальный размер класса
    max_class_size = np.max(class_counts)

    X_balanced = []
    y_balanced = []

    for class_label in unique_classes:
        # Выбираем индексы, соответствующие текущему классу
        class_indices = np.where(y == class_label)[0]

        # Если класс недостающий — нужно дополнить
        if len(class_indices) < max_class_size:
            # Количество элементов, которые нужно дополнить
            num_to_add = max_class_size - len(class_indices)
            # Случайным образом выбираем элементы для дополнения
            indices_to_add = np.random.choice(class_indices, size=num_to_add, replace=True)

            # Преобразуем элементы (например, случайный угол поворота)
            X_to_add = X[indices_to_add]  # Выбираем нужные элементы
            X_to_add = augment_data(X_to_add)  # Аугментация данных (например, поворот)

            # Добавляем в сбалансированный датасет
            X_balanced.append(np.concatenate([X[class_indices], X_to_add]))
            y_balanced.append(np.concatenate([y[class_indices], np.full(num_to_add, class_label)]))

        # Если класс превышает размер — обрезаем
        else:
            indices_to_keep = np.random.choice(class_indices, size=max_class_size, replace=False)
            X_balanced.append(X[indices_to_keep])
            y_balanced.append(y[indices_to_keep])

    # Объединяем все классы в итоговые массивы
    X_balanced = np.concatenate(X_balanced)
    y_balanced = np.concatenate(y_balanced)

    return X_balanced, y_balanced


def augment_data(X_batch):
    """Аугментация 3D-точек — случайный поворот в плоскости XY (вокруг оси Z)"""
    augmented_points = []

    for points in X_batch:
        # Случайный угол поворота вокруг оси Z (в градусах)
        angle_z = np.random.uniform(-30, 30)

        # Преобразуем угол в радианы для тригонометрических функций
        angle_z_rad = np.deg2rad(angle_z)

        # Матрица поворота вокруг оси Z (вращение в плоскости XY)
        rotation_matrix_z = np.array([
            [np.cos(angle_z_rad), -np.sin(angle_z_rad), 0],
            [np.sin(angle_z_rad), np.cos(angle_z_rad), 0],
            [0, 0, 1]
        ])

        # Применяем поворот ко всем точкам
        rotated_points = np.dot(points, rotation_matrix_z.T)  # Применяем матрицу поворота

        augmented_points.append(rotated_points)

    return np.array(augmented_points)


def normalized_data(data):
    # Центрирование и масштабирование для каждого облака точек
    data_normalized = np.zeros_like(data)
    data_size = data.shape[0]
    for i in range(data_size):
        cloud = data[i]  # Текущее облако точек
        cloud_centered = cloud - np.mean(cloud, axis=0)  # Центрирование относительно центра масс
        max_distance = np.max(np.linalg.norm(cloud_centered, axis=1))  # Максимальное расстояние от центра до точки
        data_normalized[i] = cloud_centered / max_distance  # Масштабирование
    return data_normalized