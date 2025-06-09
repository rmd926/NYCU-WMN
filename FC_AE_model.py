import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

#additional residual block
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        residual = x  # 保存輸入作為殘差
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual  # 殘差連接
        out = self.relu(out)
        return out


class FC_AE(nn.Module):
    def __init__(self, input_dim, dropout_rate):
        super(FC_AE, self).__init__()
        
        # Encoder: 壓縮特徵
        self.fc1 = nn.Linear(input_dim, 256)
        self.res_block1 = ResidualBlock(256, 128, dropout_rate)
        self.res_block2 = ResidualBlock(256, 128, dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        
        # Decoder: 還原特徵
        self.res_block3 = ResidualBlock(128, 64, dropout_rate)
        self.res_block4 = ResidualBlock(128, 64, dropout_rate)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 1)  # 輸出層
        
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder 部分
        x = self.relu(self.fc1(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.relu(self.fc2(x))
        
        # Decoder 部分
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # 輸出層
        return x
