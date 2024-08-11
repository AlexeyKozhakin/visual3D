from pyntcloud import PyntCloud
import pandas as pd
import transfom_to_model_data as tmd
def data_loader_ply(ply_file_paths):
    list_cloud_datasets = [PyntCloud.from_file(ply_file_path).points
                      for ply_file_path in ply_file_paths]
    df = pd.concat(list_cloud_datasets, ignore_index=True)
    return df

paths = '../data/Toronto_3D/'
ply_files = ['L001.ply', 'L002.ply', 'L003.ply']  # Замените на ваши пути
ply_file_paths = [paths + file for file in ply_files]
df_train_val = data_loader_ply(ply_file_paths)
print(df_train_val.shape)
ply_files = ['L004.ply']  # Замените на ваши пути
ply_file_paths = [paths + file for file in ply_files]
df_test = data_loader_ply(ply_file_paths)

data_train_val, cloud_labels_train_val = tmd.transorm_to_model_data(
                                         df_train_val,
                                         num_points_per_cloud=1024,
                                         batch_size=8)

print(data_train_val)