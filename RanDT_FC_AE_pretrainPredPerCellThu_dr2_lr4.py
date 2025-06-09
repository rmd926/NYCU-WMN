import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datetime
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
from FC_AE_model import FC_AE
from sklearn.model_selection import train_test_split
from torchviz import make_dot


# Loss function definition
def normalized_mse_loss(input, target):
    mse_loss = F.mse_loss(input, target)
    norm_factor = torch.mean(target ** 2)
    nmsl = mse_loss / norm_factor
    return nmsl

# 自定義取batch方式，每次取連續batch_size筆的資料、而非隨機batch_size筆資料更新梯度
class SequentialDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]



input_dim = 17 # per cell Num of asso. UE, SINR0~7 distribution, RSRP0~7 distribution
N_epochs = 50

batch_size=256
dp_rate=0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

combined_train_data = np.empty((0, 17))
combined_train_label = np.empty((0, 1))

#sce_ID_str_vec=['2'] # Less data to quickly run the sample code
sce_ID_str_vec=['1','21','22','2','3'] # More data for training & testing

for sce_ID in sce_ID_str_vec:
    print('scenario ',sce_ID)
    temp_Train_data_all = genfromtxt('sce'+str(sce_ID)+'_train_data_v2.1.8_DDT111.csv',delimiter=",")
    temp_Train_label_all = genfromtxt('sce'+str(sce_ID)+'_train_label_v2.1.8_DDT111.csv',delimiter=",")


    cell1_train_data = temp_Train_data_all[1:,0:17]
    cell2_train_data = temp_Train_data_all[1:,17:34]
    cell3_train_data = temp_Train_data_all[1:,34:51]
    cell4_train_data = temp_Train_data_all[1:,51:68]
    cell5_train_data = temp_Train_data_all[1:,68:85]


    print('===================')
    combined_train_data = np.vstack((combined_train_data, cell1_train_data))
    combined_train_data = np.vstack((combined_train_data, cell2_train_data))
    combined_train_data = np.vstack((combined_train_data, cell3_train_data))
    combined_train_data = np.vstack((combined_train_data, cell4_train_data))
    combined_train_data = np.vstack((combined_train_data, cell5_train_data))
    # 檢查新矩陣的尺寸
    print(combined_train_data.shape)
    print('===================')


    cell1_train_label = temp_Train_label_all[1:,0].reshape(-1, 1)
    cell2_train_label = temp_Train_label_all[1:,1].reshape(-1, 1)
    cell3_train_label = temp_Train_label_all[1:,2].reshape(-1, 1)
    cell4_train_label = temp_Train_label_all[1:,3].reshape(-1, 1)
    cell5_train_label = temp_Train_label_all[1:,4].reshape(-1, 1)


    print('===================')
    combined_train_label = np.vstack((combined_train_label, cell1_train_label))
    combined_train_label = np.vstack((combined_train_label, cell2_train_label))
    combined_train_label = np.vstack((combined_train_label, cell3_train_label))
    combined_train_label = np.vstack((combined_train_label, cell4_train_label))
    combined_train_label = np.vstack((combined_train_label, cell5_train_label))
    print("combined_train_label.shape:",combined_train_label.shape)
    print('===================')


# generate abd split training and testing data
train_data, test_data, train_label, test_label = train_test_split(
    combined_train_data, combined_train_label, test_size=0.2, random_state=42)

# 檢查數據的形狀
print(f"Train data shape: {train_data.shape}, Train label shape: {train_label.shape}")
print(f"Test data shape: {test_data.shape}, Test label shape: {test_label.shape}")

N_training_data = train_data.shape[0]
N_testing_data = test_data.shape[0]
print('train_data.shape=',train_data.shape)
print('test_data.shape=',test_data.shape)

print('Finsh training data loading')

targets = torch.tensor(train_label, dtype=torch.float32)
data = torch.tensor(train_data, dtype=torch.float32)

V_thu = torch.tensor(test_label, dtype=torch.float32)
V_data = torch.tensor(test_data, dtype=torch.float32)

print(f"Train data shape: {data.shape}, Train label shape: {targets.shape}")
print(f"Test data shape: {V_data.shape}, Test label shape: {V_thu.shape}")


print('/////////////////////')
print('data.shape=',data.shape)
print('/////////////////////')
dataset = SequentialDataset(data, targets)




def collate_fn(batch):
    data_batch = torch.stack([torch.from_numpy(item[0]) if isinstance(item[0], np.ndarray) else item[0] for item in batch])
    target_batch = torch.stack([torch.from_numpy(item[1]) if isinstance(item[1], np.ndarray) else item[1] for item in batch])
    return data_batch, target_batch


train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)


val_dataset = SequentialDataset(V_data, V_thu)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=0)

train_loss_per_epoch = np.zeros(N_epochs)
valid_loss_per_epoch = np.zeros(N_epochs)




def train(model, train_loader, valid_loader, N_epochs, batch_size=batch_size, learning_rate=1e-4):
    print('train(), N_epochs=', N_epochs)
    training_loss = np.zeros(N_epochs)
    testing_loss = np.zeros(N_epochs)
    
    # 定義優化器和學習率調度器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Early Stopping 初始化
    early_stop_patience = 10
    early_stop_counter = 0
    best_test_loss = float('inf')

    for epoch in range(N_epochs):
        train_loss = 0
        test_loss = 0
        model.train()
        
        # 訓練階段
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = normalized_mse_loss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader.dataset)
        training_loss[epoch] = train_loss

        # 驗證階段
        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                t_loss = normalized_mse_loss(output, target)
                test_loss += t_loss.item()

        test_loss /= len(valid_loader.dataset)
        testing_loss[epoch] = test_loss

        # 更新學習率
        scheduler.step(test_loss)

        # Early Stopping 檢查
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            early_stop_counter = 0  # 重置計數器
        else:
            early_stop_counter += 1

        print(f'Epoch [{epoch + 1}/{N_epochs}], Train Loss: {train_loss:.8f}; Test Loss: {test_loss:.8f}, at {datetime.datetime.now():%Y-%m-%d %H:%M:%S}')

        # 如果達到 Early Stopping 條件，結束訓練
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            break

    return training_loss[:epoch+1], testing_loss[:epoch+1]


# 宣告模型
model_reg = FC_AE(input_dim,dp_rate)
model_reg = model_reg.to(device)

# ========================================================================
summary(model_reg)
print('Per Cell model, start to train!')
train_loss_per_epoch, valid_loss_per_epoch = train(model_reg, train_loader, valid_loader, N_epochs)
print('AE reg model training finish')

torch.save(model_reg.state_dict(), 'RanDT_FC_AE_5cell_pretrain_model_dr2_lr4.tar')
print('AE reg model torch.save OK')
# ========================================================================


# Model loading
model = FC_AE(input_dim, dp_rate)
model = model.to(device)

# Load model and evaluate
model.load_state_dict(torch.load('RanDT_FC_AE_5cell_pretrain_model_dr2_lr4.tar'))


# 測試模型
model.eval()
with torch.no_grad():
    test_data = V_data[:200,:].to(device)
    predicted_value = model(test_data)


predicted_value_cpu = predicted_value.cpu().numpy()
print('V_thu.shape=',V_thu.shape)
print('predicted_value_cpu.shape=',predicted_value_cpu.shape)

fig1=plt.figure()
fig1=plt.plot(V_thu[:200],label='Ground truth')
fig1=plt.plot(predicted_value_cpu,linestyle='dashed',label='Prediction')
fig1=plt.title('Cell DL thu pred., dr2_lr4, FC AE, Ndata='+str(N_training_data))
fig1=plt.legend()
fig1=plt.grid()
fig1=plt.xlabel('Data')
fig1=plt.ylabel('Throughput (Mbps)')
fig1=plt.savefig('DL_thu_pred_FC_AE_5UE_Bsize'+str(batch_size)+'_dr2_lr4.png')
fig1=plt.show()


fig2 = plt.figure()
fig2 = plt.semilogy(train_loss_per_epoch, label='Training Loss')
fig2 = plt.semilogy(valid_loss_per_epoch, label='Validation Loss')
fig2=plt.title('MSE FC AE, ip dim='+str(input_dim)+', Ndata='+str(N_training_data))
fig2=plt.legend()
fig2=plt.xlabel('Epochs')
fig2=plt.show()
fig2=plt.grid()
fig2=plt.savefig('DL_loss_FC_AE_5UE_Bsize'+str(batch_size)+'_dr2_lr4.png')


path_file='Thu_pretrain_results_FC_AE_5UE_Bsize'+str(batch_size)+'_dr2_lr4.npz'

np.savez(path_file,thu_label=V_thu,predicted_value_cpu=predicted_value_cpu,train_loss_per_epoch=train_loss_per_epoch,\
            valid_loss_per_epoch=valid_loss_per_epoch)
