from pyntcloud import PyntCloud
import pandas as pd
def data_loader_ply(ply_file_paths):
    list_cloud_datasets = [PyntCloud.from_file(ply_file_path).points
                      for ply_file_path in ply_file_paths]
    # Конвертация в DataFrame
    df = pd.concat(list_cloud_datasets, ignore_index=True)
    return df

ply_file_paths = [
    'L004.ply'
]


