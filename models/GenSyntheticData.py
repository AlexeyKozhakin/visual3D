import pandas as pd
import numpy as np

# Установим количество точек в датасете
def gen_syth_data(num_points=1000, num_classes=5):
    # Генерация случайных данных
    np.random.seed(0)  # Для воспроизводимости результатов
    x = np.random.rand(num_points)
    y = np.random.rand(num_points)
    z = np.random.rand(num_points)

    # Генерация случайных меток от 1 до 10
    scalar_label = np.random.randint(0, num_classes, size=num_points)

    # Создание DataFrame
    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'scalar_Label': scalar_label
    })

    # Вывод первых нескольких строк DataFrame
    print(df.head())
    return df

if __name__ == '__main__':
   gen_syth_data(num_points=1000, num_classes=5)