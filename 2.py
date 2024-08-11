import plotly.graph_objs as go
import pandas as pd
import numpy as np
from pyntcloud import PyntCloud
import plotly.colors as pc

def colors_from_labels(labels):
    # Получение списка контрастных цветов
    unique_labels = labels.unique()
    num_classes = len(unique_labels)
    colors = pc.qualitative.Set1 + pc.qualitative.Pastel1  # Используем несколько контрастных палитр

    # Создание цветового словаря: метка класса -> цвет
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}

    # Присвоение цвета каждой точке на основе ее метки класса
    point_colors = [color_map[label] for label in labels]
    return point_colors
def data_loader_ply(ply_file_paths):
    list_cloud_datasets = [PyntCloud.from_file(ply_file_path).points
                      for ply_file_path in ply_file_paths]
    # Конвертация в DataFrame
    df = pd.concat(list_cloud_datasets, ignore_index=True)
    return df

# Список путей к вашим PLY файлам
paths = 'data/Toronto_3D/'
#ply_files = ['L001.ply', 'L002.ply', 'L003.ply', 'L004.ply']  # Замените на ваши пути
ply_files = ['L002.ply','L004.ply']  # Замените на ваши пути
ply_file_paths = [paths + file for file in ply_files]

df = data_loader_ply(ply_file_paths)

dn = 200
N=df.shape[0]

# Конвертация значений RGB в строки формата 'rgb(r, g, b)'
# colors = ['rgb({}, {}, {})'.format(r, g, b) for r, g, b in zip(df['red'].iloc[:N:dn],
#                                                                df['green'].iloc[:N:dn],
#                                                                df['blue'].iloc[:N:dn])]
colors = colors_from_labels(df['scalar_Label'].iloc[:N:dn])
# Создание объекта визуализации
trace = go.Scatter3d(
    x=df['x'].iloc[:N:dn],
    y=df['y'].iloc[:N:dn],
    z=df['z'].iloc[:N:dn],
    mode='markers',
    marker=dict(
        size=5,
        color=colors,  # Указываем цвета как строки формата 'rgb(r, g, b)'
        opacity=0.8
    )
)

data = [trace]

# Функция для вращающейся анимации
def visualize_rotate(data):
    frames = []

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 2 * np.pi, 0.1):
        xe, ye, ze = rotate_z(1.25, 1.25, 1.25, t)  # Радиус поворота
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))

    fig = go.Figure(data=data,
                    layout=go.Layout(
                        scene=dict(
                            #aspectmode='cube',  # Сохраняет одинаковый масштаб по всем осям
                            aspectratio=dict(x=1, y=1, z=0.1),  # Соотношение осей 1:1:1
                        ),
                        updatemenus=[dict(type='buttons',
                                          showactive=False,
                                          y=1,
                                          x=0.8,
                                          xanchor='left',
                                          yanchor='bottom',
                                          pad=dict(t=45, r=10),
                                          buttons=[dict(label='Play',
                                                        method='animate',
                                                        args=[None, dict(frame=dict(duration=50, redraw=True),
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

# Визуализация с вращением
fig = visualize_rotate(data)
fig.show()
