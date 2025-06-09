
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from FC_AE_model import FC_AE
import torch.nn.functional as F

# ---------------------------
# Data Augmentation Functions
# ---------------------------
def add_noise(data, noise_level=0.01):
    """在數據中添加高斯噪聲"""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def random_mask(data, mask_rate=0.1):
    """隨機將數據的部分特徵置為零"""
    mask = np.random.choice([0, 1], size=data.shape, p=[mask_rate, 1 - mask_rate])
    return data * mask

def scale_data(data, scale_range=(0.9, 1.1)):
    """輕微縮放數據"""
    scale = np.random.uniform(scale_range[0], scale_range[1], size=(data.shape[0], 1))
    return data * scale

# ---------------------------
# Dataset with Augmentation
# ---------------------------
class FineTuneDataset(Dataset):
    def __init__(self, data, targets, augment=False):
        self.data = data
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        target = self.targets[idx]

        # 應用數據增強
        if self.augment:
            data_point = add_noise(data_point, noise_level=0.01)
            data_point = random_mask(data_point, mask_rate=0.1)
            data_point = scale_data(data_point)

        return torch.tensor(data_point, dtype=torch.float32), torch.tensor(target, dtype=torch.float32).squeeze()

# ---------------------------
# 自定義 normalized MSE 損失函數
# ---------------------------
def normalized_mse_loss(input, target):
    input = input.view(target.shape)  # 確保輸入與目標形狀匹配
    mse_loss = F.mse_loss(input, target)
    norm_factor = torch.mean(target ** 2)
    nmsl = mse_loss / norm_factor
    return nmsl

# ---------------------------
# Fine-Tune 訓練邏輯
# ---------------------------
def fine_tune_model(model, train_loader, val_loader, device, N_epochs=300, learning_rate=1e-3):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_patience = 10

    train_loss_list, val_loss_list = [], []

    for epoch in range(N_epochs):
        model.train()
        train_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = normalized_mse_loss(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = normalized_mse_loss(output, target)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_loss_list.append(val_loss)

        scheduler.step(val_loss)

        print(f"Epoch [{epoch+1}/{N_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'fine_tuned_model.pth')
            print(f"Best model saved at epoch {epoch+1}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    return train_loss_list, val_loss_list

# ---------------------------
# 畫圖函數
# ---------------------------
def plot_results(train_loss, val_loss, ground_truth, predictions):
    plt.figure()
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_curve.png')
    plt.show()

    plt.figure()
    plt.plot(ground_truth[:200], label='Ground Truth')
    plt.plot(predictions[:200], linestyle='--', label='Prediction')
    plt.title('Throughput Prediction (First 200 Data Points)')
    plt.xlabel('Data')
    plt.ylabel('Throughput (Mbps)')
    plt.legend()
    plt.grid()
    plt.savefig('throughput_comparison_200.png')
    plt.show()

# ---------------------------
# Main Script
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加載數據
    fine_tune_data = np.random.rand(1000, 17)  # 模擬數據
    fine_tune_labels = np.random.rand(1000, 1)  # 模擬標籤
    train_data, val_data, train_labels, val_labels = train_test_split(
        fine_tune_data, fine_tune_labels, test_size=0.25, random_state=42
    )

    # 建立 Dataset 和 DataLoader
    train_dataset = FineTuneDataset(train_data, train_labels, augment=True)
    val_dataset = FineTuneDataset(val_data, val_labels, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # 初始化模型
    input_dim = 17
    dp_rate = 0.2
    model = FC_AE(input_dim, dp_rate)
    model.load_state_dict(torch.load('Pretrained_weight.tar'))
    model = model.to(device)

    # 凍結特定層
    freeze_layers = ["res_block1", "res_block3"]
    for name, param in model.named_parameters():
        if any(layer in name for layer in freeze_layers):
            param.requires_grad = False
            print(f"Layer {name} frozen.")

    print("Starting Fine-Tuning...")
    train_loss_list, val_loss_list = fine_tune_model(model, train_loader, val_loader, device, N_epochs=300, learning_rate=1e-4)
    print("Fine-Tuning Complete!")

    # 測試模型並獲得預測值
    model.load_state_dict(torch.load('fine_tuned_model.pth'))
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(val_data, dtype=torch.float32).to(device)).cpu().numpy()

    # 畫圖
    plot_results(train_loss_list, val_loss_list, val_labels.flatten(), predictions.flatten())
