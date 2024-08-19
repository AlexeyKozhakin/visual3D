from torch.utils.data import DataLoader, TensorDataset

import torch
import torch.nn as nn
import torch.optim as optim
from PointNet import PointNet

def calculate_accuracy(predictions, labels):
    correct = torch.sum(predictions == labels).item()
    total = labels.size(0)
    return correct / total


def start_training(X_train=[], y_train=[], X_val=[], y_val=[], num_batches_train=0, num_batches_val=0, batch_size=0, num_classes=0):
    # Проверка доступности GPU и перемещение модели на GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print('batch size:', batch_size)
    print('num_batches_train:', num_batches_train)
    print('num_batches_val:', num_batches_val)

    # Создание модели и перемещение на GPU
    model = PointNet(classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Подготовка данных и DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Обучение модели
    model.train()
    num_epochs = 2

    # Списки для хранения значений потерь и точности
    epochs_train_list = []
    losses_train_list = []
    accuracies_train_list = []

    epochs_val_list = []
    losses_val_list = []
    accuracies_val_list = []

    for epoch in range(num_epochs):
        for i, (X_train_batch, y_train_batch) in enumerate(train_loader):
            optimizer.zero_grad()

            # Перемещение данных на устройство
            X_train_batch = X_train_batch.permute(0, 2, 1).to(device)
            y_train_batch = y_train_batch.to(device)

            # Прямой проход через модель
            output, _, _ = model(X_train_batch)

            # Получение предсказанных меток
            _, predicted_train = torch.max(output, 1)

            # Вычисление функции потерь
            loss_train = criterion(output, y_train_batch)

            # Вычисление точности
            accuracy_train = calculate_accuracy(predicted_train, y_train_batch)

            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss_train.item()}, Accuracy: {accuracy_train:.4f}')

            # Обратный проход и оптимизация
            loss_train.backward()
            optimizer.step()

            # Сохранение данных для графиков
            epochs_train_list.append(epoch + (i / len(train_loader)))
            losses_train_list.append(loss_train.item())
            accuracies_train_list.append(accuracy_train)

        # Оценка на валидационной выборке
        model.eval()  # Переключение в режим оценки
        with torch.no_grad():
            for i, (X_val_batch, y_val_batch) in enumerate(val_loader):
                X_val_batch = X_val_batch.permute(0, 2, 1).to(device)
                y_val_batch = y_val_batch.to(device)

                # Прямой проход через модель
                output, _, _ = model(X_val_batch)

                # Получение предсказанных меток
                _, predicted_val = torch.max(output, 1)

                # Вычисление функции потерь
                loss_val = criterion(output, y_val_batch)

                # Вычисление точности
                accuracy_val = calculate_accuracy(predicted_val, y_val_batch)

                print(f'Epoch [{epoch+1}/{num_epochs}], Validation Batch [{i+1}/{len(val_loader)}], Loss: {loss_val.item()}, Accuracy: {accuracy_val:.4f}')

                # Сохранение данных для графиков
                epochs_val_list.append(epoch + (i / len(val_loader)))
                losses_val_list.append(loss_val.item())
                accuracies_val_list.append(accuracy_val)
        model.train()  # Возвращаемся в режим тренировки

    # Построение графиков
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_train_list, losses_train_list, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.plot(epochs_val_list, losses_val_list, marker='o', linestyle='-', color='r', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_train_list, accuracies_train_list, marker='o', linestyle='-', color='b', label='Train Accuracy')
    plt.plot(epochs_val_list, accuracies_val_list, marker='o', linestyle='-', color='r', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Сохранение модели
    torch.save(model.state_dict(), 'model.pth')
