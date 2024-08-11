#With Legends
#visual predict test
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from pyntcloud import PyntCloud
import plotly.colors as pc

def colors_from_labels(labels):
    unique_labels = labels.unique()
    num_classes = len(unique_labels)
    colors = pc.qualitative.Set1 + pc.qualitative.Pastel1  # Используем несколько контрастных палитр

    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    return color_map

def data_loader_ply(ply_file_paths):
    list_cloud_datasets = [PyntCloud.from_file(ply_file_path).points
                      for ply_file_path in ply_file_paths]
    df = pd.concat(list_cloud_datasets, ignore_index=True)
    return df

paths = 'data/Toronto_3D/'
ply_files = ['L002.ply']  # Замените на ваши пути
ply_file_paths = [paths + file for file in ply_files]

df = data_loader_ply(ply_file_paths)
df2 = df.copy()
# Найдем разницу между максимальным и минимальным значениями в df
shift_value = df['x'].max() - df['x'].min()

# Смещаем значения в df2 на shift_value
df2['x'] += shift_value
df = pd.concat([df,df2], ignore_index=True)

dn = 200
N = df.shape[0]

color_map = colors_from_labels(df['scalar_Label'])
data = []
classes_list = [
    'Unclassified',
    'Ground',
    'Road_markings',
    'Natural',
    'Building',
    'Utility_line',
    'Pole',
    'Car',
    'Fence'
]
# Создание объекта визуализации для каждого класса
for label, color in color_map.items():
    class_mask = df['scalar_Label'] == label
    trace = go.Scatter3d(
        x=df['x'].iloc[:N:dn][class_mask],
        y=df['y'].iloc[:N:dn][class_mask],
        z=df['z'].iloc[:N:dn][class_mask],
        mode='markers',
        marker=dict(
            size=5,
            color=color,
            opacity=0.8
        ),
        name=f'{classes_list[int(label)]}'  # Имя в легенде
    )
    data.append(trace)

# Функция для вращающейся анимации
def visualize_rotate(data):
    frames = []

    def rotate_z(x, y, z, theta):
        w = x + 1j * y
        return np.real(np.exp(1j * theta) * w), np.imag(np.exp(1j * theta) * w), z

    for t in np.arange(0, 2 * np.pi, 0.1):
        xe, ye, ze = rotate_z(1.25, 1.25, 1.25, t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, y=ye, z=ze))))))

    fig = go.Figure(data=data,
                    layout=go.Layout(
                        scene=dict(
                            aspectratio=dict(x=1, y=1, z=0.1),
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
