import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from PointNet import PointNet

def calculate_accuracy(predictions, labels):
    correct = torch.sum(predictions == labels).item()
    total = labels.size(0)
    return correct / total

def start_training(X_train=[], y_train=[], X_val=[], y_val=[],
                   num_batches_train=0, num_batches_val=0, batch_size=0, num_classes=0):
    # Проверка доступности GPU и перемещение модели на GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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

    # Списки для хранения значений потерь и точности по эпохам
    epoch_train_loss_list = []
    epoch_train_accuracy_list = []

    epoch_val_loss_list = []
    epoch_val_accuracy_list = []

    for epoch in range(num_epochs):
        # Инициализация для накопления значений в течение эпохи
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

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

            # Обратный проход и оптимизация
            loss_train.backward()
            optimizer.step()

            # Накопление потерь и правильных предсказаний
            total_train_loss += loss_train.item() * X_train_batch.size(0)
            total_train_correct += torch.sum(predicted_train == y_train_batch).item()
            total_train_samples += X_train_batch.size(0)

        # Усреднение потерь и точности по эпохе
        avg_train_loss = total_train_loss / total_train_samples
        avg_train_accuracy = total_train_correct / total_train_samples

        # Сохранение данных для графиков
        epoch_train_loss_list.append(avg_train_loss)
        epoch_train_accuracy_list.append(avg_train_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}')

        # Оценка на валидационной выборке
        model.eval()  # Переключение в режим оценки
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0

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

                # Накопление потерь и правильных предсказаний
                total_val_loss += loss_val.item() * X_val_batch.size(0)
                total_val_correct += torch.sum(predicted_val == y_val_batch).item()
                total_val_samples += X_val_batch.size(0)

        # Усреднение потерь и точности по эпохе
        avg_val_loss = total_val_loss / total_val_samples
        avg_val_accuracy = total_val_correct / total_val_samples

        # Сохранение данных для графиков
        epoch_val_loss_list.append(avg_val_loss)
        epoch_val_accuracy_list.append(avg_val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}')

        model.train()  # Возвращаемся в режим тренировки

    # Построение графиков
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), epoch_train_loss_list, marker='o', linestyle='-', color='b', label='Train Loss')
    plt.plot(range(1, num_epochs + 1), epoch_val_loss_list, marker='o', linestyle='-', color='r', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), epoch_train_accuracy_list, marker='o', linestyle='-', color='b', label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), epoch_val_accuracy_list, marker='o', linestyle='-', color='r', label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Сохранение модели
    torch.save(model.state_dict(), 'model.pth')
