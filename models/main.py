import torch
from loader import get_data_for_model
from training_models2 import start_training

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

num_points_train = 100_000
num_points_val = 50_000
batch_size = 10
num_points_per_cloud = 1024
num_classes = 9

num_batches_train = num_points_train//(num_points_per_cloud*batch_size)
num_batches_val = num_points_val//(num_points_per_cloud*batch_size)
print('number of batches train: ', num_batches_train)
print('number of batches val: ', num_batches_val)


(data_train_val, cloud_labels_train_val,
 data_test, cloud_labels_test) = get_data_for_model(synthetic = False,
                                                    num_points_train = num_points_train,
                                                    num_points_val = num_points_val,
                                                    num_points_per_cloud=num_points_per_cloud,
                                                    num_classes = num_classes,
                                                    batch_size=batch_size)

print(data_train_val.shape)
num_points_train = data_train_val.shape[0]
num_points_val = data_test.shape[0]
batch_size = 20

num_batches_train = num_points_train//(batch_size)
num_batches_val = num_points_val//(batch_size)


start_training(X_train=data_train_val, y_train=cloud_labels_train_val,
                X_val=data_test, y_val=cloud_labels_test,
                num_batches_train=num_batches_train, num_batches_val=num_batches_val, batch_size=batch_size, num_classes=num_classes)


