import open3d as o3d
import numpy as np

# Список путей к вашим PLY файлам
paths = 'data/Toronto_3D/'
ply_files = ['L001.ply', 'L002.ply', 'L003.ply', 'L004.ply']  # Замените на ваши пути
ply_files = [paths + file for file in ply_files]
#ply_files = ['data/Toronto_3D/L002.ply']
# Создаем пустой объект PointCloud для объединенного облака точек
combined_pcd = o3d.geometry.PointCloud()

# Загрузка и объединение облаков точек из всех файлов
for ply_file_path in ply_files:
    # Загрузка облака точек
    pcd = o3d.io.read_point_cloud(ply_file_path)

    # Проверка успешности загрузки
    if not pcd.is_empty():
        print(f"Successfully loaded the point cloud from {ply_file_path}.")
        combined_pcd += pcd
    else:
        print(f"Failed to load the point cloud from {ply_file_path}.")

# Прореживание объединенного облака точек (опционально)
step = 500
if len(combined_pcd.points) > step:
    indices = np.arange(0, len(combined_pcd.points), step)
    downsampled_combined_pcd = combined_pcd.select_by_index(indices)
else:
    downsampled_combined_pcd = combined_pcd

# Визуализация объединенного облака точек
o3d.visualization.draw_geometries([downsampled_combined_pcd])

# Сохранение объединенного облака точек (опционально)
o3d.io.write_point_cloud("combined_cloud.ply", downsampled_combined_pcd)
