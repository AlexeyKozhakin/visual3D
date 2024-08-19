from pyntcloud import PyntCloud
import pandas as pd
import transfom_to_model_data as tmd
import h5py
def data_loader_ply(ply_file_paths):
    list_cloud_datasets = [PyntCloud.from_file(ply_file_path).points
                      for ply_file_path in ply_file_paths]
    df = pd.concat(list_cloud_datasets, ignore_index=True)
    return df

paths = '../data/Toronto_3D/'
#ply_files = ['L001.ply', 'L002.ply', 'L003.ply', 'L004.ply']  # Замените на ваши пути
#ply_files = ['L001.ply', 'L002.ply', 'L003.ply']  # Замените на ваши пути
ply_files = ['L004.ply']  # Замените на ваши пути

#ply_files = ['L002.ply']  # Замените на ваши пути
ply_file_paths = [paths + file for file in ply_files]
df_train_val = data_loader_ply(ply_file_paths)
print(df_train_val.shape)
# ply_files = ['L004.ply']  # Замените на ваши пути
# ply_file_paths = [paths + file for file in ply_files]
# df_test = data_loader_ply(ply_file_paths)

data_train_val, cloud_labels_train_val = tmd.transorm_to_model_data(
                                         df_train_val,
                                         num_points_per_cloud=1024,
                                         batch_size=8)

# data_test, cloud_labels_test = tmd.transorm_to_model_data(
#                                          df_test,
#                                          num_points_per_cloud=1024,
#                                          batch_size=8)

# Создание файла HDF5
#with h5py.File('../data/Toronto_3D/train_dataset_L001_L002_L003.h5', 'w') as hdf:
with h5py.File('../data/Toronto_3D/train_dataset_L004.h5', 'w') as hdf:
    # Запись тензора X в группу 'X'
    hdf.create_dataset('X', data=data_train_val)

    # Запись меток y в группу 'y'
    hdf.create_dataset('y', data=cloud_labels_train_val)

# # Создание файла HDF5
# with h5py.File('test_dataset_L004.h5', 'w') as hdf:
#     # Запись тензора X в группу 'X'
#     hdf.create_dataset('X', data=data_test)
#
#     # Запись меток y в группу 'y'
#     hdf.create_dataset('y', data=cloud_labels_test)