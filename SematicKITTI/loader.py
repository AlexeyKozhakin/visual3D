import numpy as np
import plotly.io as pio
def load_point_cloud(file_path):
    # Загружаем бинарный файл
    point_cloud = np.fromfile(file_path, dtype=np.float32)

    # У каждой точки есть 4 значения (x, y, z, интенсивность)
    point_cloud = point_cloud.reshape((-1, 4))  # Переводим в матрицу Nx4
    return point_cloud


def load_labels(file_path):
    # Загружаем файл меток в формате uint32
    labels = np.fromfile(file_path, dtype=np.uint32)

    # Извлекаем семантические метки (16 младших бит)
    semantic_labels = labels & 0xFFFF  # Операция побитового И

    # Извлекаем инстанс-метки (старшие 16 бит)
    instance_labels = labels >> 16  # Побитовый сдвиг вправо

    return semantic_labels, instance_labels


import open3d as o3d

import numpy as np
import plotly.graph_objects as go


def visualize_rotate(points, labels=None):
    # Проверяем, что входной массив имеет правильную форму
    if points.shape[1] != 3:
        raise ValueError("points array must have shape (N, 3), where N is the number of points")

    # Проверяем, что метки переданы и их длина совпадает с числом точек
    if labels is not None:
        if len(labels) != points.shape[0]:
            raise ValueError("labels array must have the same length as points array")

    frames = []

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    # Создаем кадры для анимации
    for t in np.arange(0, 2 * np.pi, 0.1):
        # Применяем вращение ко всем точкам
        x_rot, y_rot, z_rot = rotate_z(points[:, 0], points[:, 1], points[:, 2], t)

        # Цвет точек в зависимости от меток
        if labels is not None:
            colors = labels
        else:
            colors = 'rgba(0, 0, 255, 0.8)'  # Цвет по умолчанию (синий)

        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=x_rot,
                        y=y_rot,
                        z=z_rot,
                        mode='markers',
                        marker=dict(size=3, color=colors, colorscale='Viridis', colorbar=dict(title='Labels'))
                    )
                ],
                name=f'Frame {int(t * 10)}'
            )
        )

    # Создаем начальный график
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(size=3, color=colors, colorscale='Viridis', colorbar=dict(title='Labels'))
            )
        ],
        layout=go.Layout(
            scene=dict(
                aspectratio=dict(x=1, y=1, z=1),
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    y=1,
                    x=0.8,
                    xanchor='left',
                    yanchor='bottom',
                    pad=dict(t=45, r=10),
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=50, redraw=True),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode='immediate'
                            )]
                        )
                    ]
                )
            ]
        ),
        frames=frames
    )

    return fig


if __name__ == '__main__':
    # Пример использования
    path = (r'C:\Users\alexe\Downloads\UM'
                 r'\work\Data\Semantic KITTI\Semantic KITTI'
                 r'\dataset\sequences\00')
    bin_file = path + r'\velodyne\000000.bin'
    point_cloud = load_point_cloud(bin_file)
    print(point_cloud.shape)  # Должно быть (N, 4), где N — количество точек

    # Пример использования
    label_file = path + r'\labels\000000.label'
    semantic_labels, instance_labels = load_labels(label_file)
    print(semantic_labels.shape)  # Должно быть (N,), где N — количество точек
    print(set(semantic_labels.tolist()))
    print(set(instance_labels.tolist()))
    # Пример использования
    # fig = visualize_rotate(point_cloud[:,:3], semantic_labels)
    # # Отображение в браузере
    # pio.show(fig)
