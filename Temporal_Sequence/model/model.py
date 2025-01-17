import torch.nn as nn
import torch

class simple_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(simple_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(1000)
        self.bn2 = nn.BatchNorm1d(100)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        # x = self.output(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1600)
        self.fc4 = nn.Linear(1600, 400)
        self.fc5 = nn.Linear(400, 100)
        self.fc6 = nn.Linear(100, 10)
        self.fc7 = nn.Linear(10,output_dim)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(1600)
        self.bn4 = nn.BatchNorm1d(400)
        self.bn5 = nn.BatchNorm1d(100)
        self.bn6 = nn.BatchNorm1d(10)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = self.output(x)
        return x
    
class Deep_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Deep_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 6400)
        self.fc3 = nn.Linear(6400, 4800)
        self.fc4 = nn.Linear(4800, 3200)
        self.fc5 = nn.Linear(3200, 1600)
        self.fc6 = nn.Linear(1600, 800)
        self.fc7 = nn.Linear(800,320)
        self.fc8 = nn.Linear(320,160)
        self.fc9 = nn.Linear(160, 40)
        self.fc10 = nn.Linear(40, 10)
        self.fc11 = nn.Linear(10, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(6400)
        self.bn3 = nn.BatchNorm1d(4800)
        self.bn4 = nn.BatchNorm1d(3200)
        self.bn5 = nn.BatchNorm1d(1600)
        self.bn6 = nn.BatchNorm1d(800)
        self.bn7 = nn.BatchNorm1d(320)
        self.bn8 = nn.BatchNorm1d(160)
        self.bn9 = nn.BatchNorm1d(40)
        self.bn10 = nn.BatchNorm1d(10)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = self.bn7(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc8(x)
        x = self.bn8(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc9(x)
        x = self.bn9(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc10(x)
        x = self.bn10(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc11(x)
        x = self.output(x)

        return x

class TransformerLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_length, num_layers=1):
        super(TransformerLayer, self).__init__()
        encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4, 
            dim_feedforward=hidden_size, 
            dropout=0.1,
            batch_first=True
            )
        self.encoder = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.embedding = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=(1, input_size), stride=1)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.fc2 = nn.Linear(seq_length, output_size)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        out = self.embedding(x).squeeze().transpose(2, 1)
        out = self.encoder(out)
        out = self.fc1(out).squeeze()
        out = self.relu(out)
        out = self.fc2(out).squeeze()

        return out