from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import torch.nn as nn

class GCNModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, in_channels*2)
        self.conv2 = GCNConv(in_channels*2, in_channels*4)
        self.conv3 = GCNConv(in_channels*4, in_channels*2)

        self.fc1 = nn.Linear(21, 1024)
        self.fc2 = nn.Linear(1024, out_channels)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = x.flatten(1)
        print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x

