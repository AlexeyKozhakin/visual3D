import open3d as o3d
import pandas as pd
import numpy as np

# Пример DataFrame (замените на ваши данные)
df = pd.DataFrame({
    'x': np.random.randint(1, 10, 100),
    'y': np.random.randint(1, 10, 100),
    'z': np.random.randint(1, 10, 100),
    'red': np.random.randint(0, 256, 100),
    'green': np.random.randint(0, 256, 100),
    'blue': np.random.randint(0, 256, 100)
})

# Преобразование данных в массивы numpy
points = df[['x', 'y', 'z']].to_numpy()
colors = df[['red', 'green', 'blue']].to_numpy() / 255.0  # Масштабирование в диапазон [0, 1]

# Создание объекта PointCloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Визуализация облака точек
o3d.visualization.draw_geometries([pcd])