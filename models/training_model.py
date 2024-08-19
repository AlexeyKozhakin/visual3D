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

        # Обучение модели
        model.train()
        num_epochs = 5

        # Списки для хранения значений потерь и точности
        epochs_train_list = []
        losses_train_list = []
        accuracies_train_list = []

        epochs_val_list = []
        losses_val_list = []
        accuracies_val_list = []

        for epoch in range(num_epochs):
            for i in range(0, num_batches_train):
                optimizer.zero_grad()

                # Получение текущего батча
                X_train_batch = torch.tensor(X_train[i:i + batch_size], dtype=torch.float32).permute(0, 2, 1).to(device)
                y_train_batch = torch.tensor(y_train[i:i + batch_size], dtype=torch.long).to(device)
                # Прямой проход через модель
                output, _, _ = model(X_train_batch)
                # Получение предсказанных меток
                _, predicted_train = torch.max(output, 1)
                # Вычисление функции потерь
                loss_train = criterion(output, y_train_batch)
                # Вычисление точности
                accuracy_train = calculate_accuracy(predicted_train, y_train_batch)

                # print(predicted)
                # print(batch_labels)

                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{num_batches_train}], Loss: {loss_train.item()}, Accuracy: {accuracy_train:.4f}')
                # Обратный проход и оптимизация
                loss_train.backward()
                optimizer.step()

                # Сохранение данных для графиков
                epochs_train_list.append(epoch + (i / num_batches_train))
                losses_train_list.append(loss_train.item())
                accuracies_train_list.append(accuracy_train)
            for i in range(0, num_batches_val):
                # Получение текущего батча
                X_val_batch = torch.tensor(X_val[i:i + batch_size], dtype=torch.float32).permute(0, 2, 1).to(device)
                y_val_batch = torch.tensor(y_val[i:i + batch_size], dtype=torch.long).to(device)
                # Прямой проход через модель
                output, _, _ = model(X_val_batch)
                # Получение предсказанных меток
                _, predicted_train = torch.max(output, 1)
                # Вычисление функции потерь
                loss_train = criterion(output, y_val_batch)
                # Вычисление точности
                accuracy_train = calculate_accuracy(predicted_train, y_val_batch)

                # print(predicted)
                # print(batch_labels)

                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{num_batches_val}], Loss: {loss_train.item()}, Accuracy: {accuracy_train:.4f}')
                # Обратный проход и оптимизация

                # Сохранение данных для графиков
                epochs_val_list.append(epoch + (i / num_batches_val))
                losses_val_list.append(loss_train.item())
                accuracies_val_list.append(accuracy_train)

            # epochs_val_list.append(epoch)
            # losses_val_list.append(0)
            # accuracies_val_list.append(0)
            # for i in range(0, num_batches_val):
            #     X_val_batch = torch.tensor(X_val[i:i + batch_size], dtype=torch.float32).permute(0, 2, 1).to(device)
            #     y_val_batch = torch.tensor(y_val[i:i + batch_size], dtype=torch.long).to(device)
            #
            #     # Прямой проход через модель
            #     output, _, _ = model(X_val_batch)
            #     # Получение предсказанных меток
            #     _, predicted_val = torch.max(output, 1)
            #     # Вычисление функции потерь
            #     loss_val = criterion(output, y_val_batch)
            #     accuracy_val = calculate_accuracy(predicted_val, y_val_batch)
            #
            #     losses_val_list[-1]+=loss_val.item()
            #     accuracies_val_list[-1]+=accuracy_val
            # losses_val_list[-1] /= batch_size
            # accuracies_val_list[-1] /= batch_size
            #print(f'Epoch [{epoch + 1}/{num_epochs}], Val Loss: {losses_val_list[-1]}, Val Accuracy: {accuracies_val_list[-1]:.4f}')
            # Сохранение данных для графиков


        # Построение графика потерь
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_train_list, losses_train_list, marker='o', linestyle='-', color='b', label='Loss')
        plt.plot(epochs_val_list, losses_val_list, marker='o', linestyle='-', color='r', label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss over Epochs')
        plt.legend()
        plt.grid(True)

        # Построение графика точности
        plt.subplot(1, 2, 2)
        plt.plot(epochs_train_list, accuracies_train_list, marker='o', linestyle='-', color='b', label='Accuracy')
        plt.plot(epochs_val_list, accuracies_val_list, marker='o', linestyle='-', color='r', label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy over Epochs')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # Сохранение всей модели
        torch.save(model, 'model.pth')


