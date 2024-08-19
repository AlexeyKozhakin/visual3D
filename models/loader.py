from pyntcloud import PyntCloud
import pandas as pd
import transfom_to_model_data as tmd
from GenSyntheticData import gen_syth_data
from imblearn.over_sampling import RandomOverSampler

def balance_dataset(df, target = 'scalar_Label'):
    # Выделяем признаки и целевой столбец
    X = df.drop(target, axis=1)
    y = df[target]

    # Инициализируем RandomOverSampler
    ros = RandomOverSampler(random_state=42)

    # Выполняем oversampling
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # Объединяем признаки и целевой столбец в новый DataFrame
    df_resampled = pd.concat(
        [pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[target])], axis=1)
    return df_resampled

def data_loader_ply(ply_file_paths):
    list_cloud_datasets = [PyntCloud.from_file(ply_file_path).points
                      for ply_file_path in ply_file_paths]
    df = pd.concat(list_cloud_datasets, ignore_index=True)
    return df
def get_data_for_model(synthetic = True, num_points_train = 10_000,
                       num_points_val = 3000, num_points_per_cloud = 1024, batch_size=8, num_classes = 5):
    if synthetic:
            df_train_val=gen_syth_data(num_points=num_points_train, num_classes=num_classes)
            df_test=gen_syth_data(num_points=num_points_val, num_classes=num_classes)
    else:
            paths = '../datasets/Toronto_3D/'
            #ply_files = ['L001.ply', 'L002.ply', 'L003.ply']  # Замените на ваши пути
            ply_files = ['L004.ply']  # Замените на ваши пути
            ply_file_paths = [paths + file for file in ply_files]
            df_train_val = data_loader_ply(ply_file_paths)
            df_train_val = df_train_val.iloc[:df_train_val.shape[0]//50, :]
            df_train_val=balance_dataset(df_train_val, target='scalar_Label')
            print(df_train_val.shape)
            ply_files = ['L001.ply']  # Замените на ваши пути
            ply_file_paths = [paths + file for file in ply_files]
            df_test = data_loader_ply(ply_file_paths)
            df_test = df_test.iloc[:df_test.shape[0]//50, :]
            print(df_test.shape)
            df_test = balance_dataset(df_test, target='scalar_Label')
            #df_train_val = df_train_val.iloc[:df_test.shape[0],:]

    data_train_val, cloud_labels_train_val = tmd.transorm_to_model_data(
                                             df_train_val,
                                             num_points_per_cloud=num_points_per_cloud,
                                             batch_size=batch_size)
    data_test, cloud_labels_test = tmd.transorm_to_model_data(
                                             df_test,
                                             num_points_per_cloud=num_points_per_cloud,
                                             batch_size=batch_size)
    #print(data_train_val)
    return (data_train_val, cloud_labels_train_val,
            data_test, cloud_labels_test)

if __name__ == '__main__':
    (data_train_val, cloud_labels_train_val,
     data_test, cloud_labels_test) = get_data_for_model(synthetic=False, num_classes=5)