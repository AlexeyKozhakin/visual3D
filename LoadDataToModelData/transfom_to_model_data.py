import numpy as np
# Определение меток для каждого облака точек на основе большинства
def get_majority_label(labels,num_batches):
    majority_labels = np.zeros(num_batches, dtype=int)
    for i in range(num_batches):
        majority_labels[i] = np.bincount(labels[i]).argmax()
    return majority_labels

def transorm_to_model_data(df, num_points_per_cloud=1024, batch_size = 8):
        # Допустим, у вас уже есть DataFrame df с колонками 'x', 'y', 'z' и 'scalar_Label'
        # Преобразование данных из DataFrame в массив numpy
        data = df[['x', 'y', 'z']].to_numpy()  # 3 координаты (x, y, z)
        cloud_labels = df['scalar_Label'].to_numpy().astype(int)  # Метки для точек

        total_points = data.shape[0]
        num_batches = total_points//(num_points_per_cloud*batch_size)
        num_points_cut = num_points_per_cloud*batch_size*num_batches

        # Убедитесь, что количество точек кратно num_points_per_cloud
        data = data[:num_points_cut]  # Урезаем до кратного количества точек
        cloud_labels = cloud_labels[:num_points_cut]

        # Формирование облаков точек
        data = data.reshape(batch_size*num_batches, num_points_per_cloud, 3)  # [batch_size, num_points_per_cloud, 3]
        cloud_labels = cloud_labels.reshape(batch_size*num_batches, num_points_per_cloud)  # [batch_size, num_points_per_cloud]

        # Центрирование и масштабирование для каждого облака точек
        data_normalized = np.zeros_like(data)
        for i in range(num_batches*batch_size):
            cloud = data[i]  # Текущее облако точек
            cloud_centered = cloud - np.mean(cloud, axis=0)  # Центрирование относительно центра масс
            max_distance = np.max(np.linalg.norm(cloud_centered, axis=1))  # Максимальное расстояние от центра до точки
            data_normalized[i] = cloud_centered / max_distance  # Масштабирование

        # Преобразование данных в тензоры
        # data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
        # cloud_labels_tensor = torch.tensor(cloud_labels, dtype=torch.long)

        # Определение меток для облаков точек на основе большинства
        cloud_labels = get_majority_label(cloud_labels, num_batches*batch_size)  # Метки для облаков точек
        return data_normalized, cloud_labels


