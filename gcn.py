import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCN(nn.Module):

    def __init__(self, adj: torch.Tensor, num_class: int):

        super(SimpleGCN, self).__init__()

        self.adj = adj.unsqueeze(0)

        self.layer_1 = nn.Conv1d(1, 4, 1)
        self.bn1 = nn.BatchNorm1d(4)
        
        self.layer_2 = nn.Conv1d(4, 8, 1)
        self.bn2 = nn.BatchNorm1d(8)
        
        self.layer_3 = nn.Conv1d(8, 8, 1)
        self.bn3 = nn.BatchNorm1d(8)
        
        self.classifier = nn.Conv1d(8, num_class, 1)

        self.relu = nn.Tanh()

    def forward(self, x):
        """
        input: x: tensor, size(batch_size, 1, num_nodes)
        """
        # ======== layer 1 ========
        x = torch.matmul(x, self.adj) # A*X
        x = self.layer_1(x) # X*W
        x = self.bn1(x)
        x = self.relu(x)

        # ======== layer 2 ========
        x = torch.matmul(x, self.adj) # A*X
        x = self.layer_2(x) # X*W
        x = self.bn2(x)
        x = self.relu(x)

        # ======== layer 3 ========
        x = torch.matmul(x, self.adj) # A*X
        x = self.layer_3(x) # X*W
        x = self.bn3(x)
        x = self.relu(x)

        x = self.classifier(x)

        return F.log_softmax(x, dim=1)