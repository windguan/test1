import torch
import torch.nn as nn
import torch.nn.functional as F


# ================= GCN 层定义 =================
class GCNLayer(nn.Module):
    """
    图卷积网络层 (Graph Convolutional Network)
    虽然我们当前数据是蛋白对拼接特征，但保留标准GCN结构供后续扩展
    """

    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x, adj=None):
        # x: (batch_size, in_features)
        # 这里我们简化为对特征做线性变换（适配当前蛋白对分类任务）
        out = self.linear(x)
        out = F.relu(out)
        return out


# ================= GAT 层定义 =================
class GATLayer(nn.Module):
    """
    图注意力网络层 (Graph Attention Network)
    """

    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.linear = nn.Linear(in_features, out_features)
        self.attention = nn.Linear(2 * out_features, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    def forward(self, x):
        # 简化版GAT：对拼接特征做带注意力的变换
        h = self.linear(x)

        # 这里我们模拟注意力机制
        attention_coeffs = torch.sigmoid(self.attention(torch.cat([h, h], dim=1)))
        h_prime = attention_coeffs * h

        return F.elu(h_prime)


# ================= 完整的 PPI 预测模型 =================
class PPIPredictor(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, model_type='gcn'):
        """
        input_dim: 输入特征维度 (两个蛋白拼接: 128+128=256)
        hidden_dim: 隐藏层维度
        model_type: 'gcn' 或 'gat'
        """
        super(PPIPredictor, self).__init__()
        self.model_type = model_type

        # 第一层
        if model_type == 'gcn':
            self.layer1 = GCNLayer(input_dim, hidden_dim)
        elif model_type == 'gat':
            self.layer1 = GATLayer(input_dim, hidden_dim)
        else:
            raise ValueError("model_type must be 'gcn' or 'gat'")

        # 第二层
        if model_type == 'gcn':
            self.layer2 = GCNLayer(hidden_dim, hidden_dim // 2)
        elif model_type == 'gat':
            self.layer2 = GATLayer(hidden_dim, hidden_dim // 2)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 2)  # 二分类: 0(不互作) / 1(互作)
        )

    def forward(self, x):
        # x shape: (batch_size, 256)
        out = self.layer1(x)
        out = F.dropout(out, p=0.5, training=self.training)
        out = self.layer2(out)
        out = self.classifier(out)
        return out